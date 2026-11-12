"""Bounded in-process diagnostics; inactive until explicitly started.

No CUDA operations, tensor reads, model changes or profiler-time disk writes.
Elapsed Python calls and stack samples are not estimates of GPU idle time.
"""
from collections import Counter
import functools
import inspect
import math
import sys
import threading
import time


class HostProfile:
    def __init__(self, runner, seconds=20, hz=50):
        if not (math.isfinite(seconds) and 0.05 <= seconds <= 60):
            raise ValueError('seconds must be finite and in [0.05,60]')
        if not (isinstance(hz,int) and 1 <= hz <= 100):
            raise ValueError('hz must be an integer in [1,100]')
        self.runner=runner;self.seconds=seconds;self.hz=hz
        self.events=[];self.samples=[];self.originals=[];self.sources={}
        self.lock=threading.Lock();self.active=threading.Event();self.finished=threading.Event()
        self.main_thread_id=threading.get_ident()
        self.started_epoch=None;self.ended_epoch=None;self.thread=None
        self.error=None

    def _wrap(self, obj, name, label):
        original=getattr(obj,name)
        if not callable(original):raise TypeError(label)
        had_own=name in vars(obj);own=vars(obj).get(name)
        self.originals.append((obj,name,had_own,own))
        function=getattr(original,'__func__',original)
        self.sources[label]=dict(file=inspect.getsourcefile(function),
                                name=getattr(function,'__qualname__',name))
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            if not self.active.is_set():return original(*args,**kwargs)
            begin=time.perf_counter_ns();error=None
            try:return original(*args,**kwargs)
            except BaseException as exc:
                error=type(exc).__name__;raise
            finally:
                end=time.perf_counter_ns()
                event=dict(phase=label,start_ns=begin,end_ns=end,
                           thread_id=threading.get_ident(),error=error)
                if label=='ple.launch':
                    # Arguments are scalar sizes; never copy token IDs or tensors.
                    event['num_reqs']=kwargs.get('num_reqs',args[0] if args else None)
                    event['num_tokens']=kwargs.get('num_tokens',args[1] if len(args)>1 else None)
                with self.lock:self.events.append(event)
        setattr(obj,name,wrapped)

    def start(self):
        if self.thread is not None:raise RuntimeError('Already started')
        connector=self.runner._ple_offload_connector
        if connector is None:raise ValueError('No PLE connector')
        try:
            for name in ('execute_model','sample_tokens'):
                self._wrap(self.runner,name,'runner.'+name)
            for name,label in [('_launch','ple.launch'),('_copy_cuda_inputs','ple.copy_inputs'),
                               ('_process_request','ple.process_request')]:
                self._wrap(connector,name,label)
            self.started_epoch=time.time();self.started_ns=time.perf_counter_ns()
            self.active.set()
            self.thread=threading.Thread(target=self._sample,name='arc3-bounded-host-profile',daemon=True)
            self.thread.start()
        except BaseException:
            self._restore();raise
        return dict(started_epoch=self.started_epoch,seconds=self.seconds,hz=self.hz,
                    main_thread_id=self.main_thread_id,phases=list(self.sources))

    def _sample(self):
        deadline=time.monotonic()+self.seconds
        try:
            while self.active.is_set() and time.monotonic()<deadline:
                timestamp=time.perf_counter_ns()
                names={t.ident:t.name for t in threading.enumerate()}
                rows=[]
                for tid,frame in sys._current_frames().items():
                    if tid==threading.get_ident():continue
                    stack=[]
                    while frame is not None and len(stack)<24:
                        code=frame.f_code
                        stack.append([code.co_filename,code.co_name,frame.f_lineno])
                        frame=frame.f_back
                    rows.append(dict(thread_id=tid,name=names.get(tid),stack=stack))
                self.samples.append(dict(time_ns=timestamp,threads=rows))
                time.sleep(min(1/self.hz,max(0,deadline-time.monotonic())))
        except BaseException as exc:self.error=repr(exc)
        finally:
            self.active.clear();self._restore();self.ended_epoch=time.time();self.finished.set()

    def _restore(self):
        for obj,name,had_own,own in reversed(self.originals):
            if had_own:setattr(obj,name,own)
            else:delattr(obj,name)
        self.originals.clear()

    def result(self):
        if not self.finished.is_set():return dict(complete=False,started_epoch=self.started_epoch)
        with self.lock:events=list(self.events)
        return dict(complete=True,started_epoch=self.started_epoch,ended_epoch=self.ended_epoch,
                    started_ns=self.started_ns,requested_seconds=self.seconds,hz=self.hz,
                    main_thread_id=self.main_thread_id,events=events,samples=list(self.samples),
                    sources=self.sources,error=self.error,
                    note='Inclusive host durations and GIL-dependent Python stack samples. They overlap GPU work and each other; not GPU idle time. Sampling can undersample long GIL-held sections. No arguments, tensors or model-visible data captured.')


def start_for_worker(worker,seconds=20,hz=50):
    previous=getattr(worker,'_arc3_host_profile',None)
    if previous is not None and not previous.finished.is_set():
        raise RuntimeError('Host profile is already active')
    profile=HostProfile(worker.model_runner,seconds,hz)
    worker._arc3_host_profile=profile
    return profile.start()


def result_for_worker(worker):
    profile=getattr(worker,'_arc3_host_profile',None)
    if profile is None:return dict(complete=False,not_started=True)
    return profile.result()
