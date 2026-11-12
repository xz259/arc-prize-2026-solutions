"""Run on pinned runtime: real FastAPI/httpx routes, fake engine/backend only.

No socket connection, model load, GPU operation, or inference request occurs.
HTTPX ASGITransport drives the real plugin routes within this Python process.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import unittest

from fastapi import FastAPI
import httpx

from persistent_endpoint import PersistentEndpoint
from session_service import SessionError, SessionManager, TurnInput, token_digest


class FakeBackend:
    def __init__(self, events):
        self.events = events
        self.submitted = []
        self.owned = set()
        self.fail_after_submit = False
        self.gate = None

    async def prepare(self, session, payload, *, initial):
        raw = payload.get('raw')
        if not isinstance(raw, dict) or not isinstance(raw.get('token_ids'), list):
            raise SessionError('Fixture requires raw token_ids', 400)
        ids = raw['token_ids']
        if any(type(x) is not int or x < 0 for x in ids):
            raise SessionError('Invalid fixture token IDs', 400)
        return TurnInput(dict(type='token', prompt_token_ids=list(ids)),
                         dict(payload.get('sampling', {})))

    async def turn(self, session, turn, *, initial):
        self.owned.add(session.session_id)
        self.submitted.append((session.session_id, list(turn.token_ids)))
        self.events.append('submit:'+session.session_id)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail_after_submit:
            raise RuntimeError('Deliberate fake engine failure')
        return dict(token_ids=[101], finish_reason='length', stop_reason=None)

    async def response(self, session, result):
        return dict(choices=[dict(index=0, token_ids=result['token_ids'],
            finish_reason=result['finish_reason'],
            message=dict(role='assistant', content='fake output'))])

    async def close(self, session):
        self.events.append('backend.close:'+session.session_id)
        self.owned.discard(session.session_id)


class TrackingManager(SessionManager):
    def __init__(self, backend, events):
        super().__init__(backend, max_context=20, max_sessions=2, turn_timeout=2)
        self.close_all_count = 0
        self.events = events

    async def close_all(self):
        self.close_all_count += 1
        self.events.append('manager.close_all')
        await super().close_all()


def start_body(sid='game', request_id='start', ids=None, **extra):
    return dict(session_id=sid, request_id=request_id,
                raw={'token_ids': [1,2] if ids is None else ids}, **extra)


class RuntimeRoutes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.backend = FakeBackend(self.events)
        self.manager = TrackingManager(self.backend, self.events)
        self.yielded_state = {'preserved_lifespan_marker': 'original-state'}

        @asynccontextmanager
        async def original_lifespan(app):
            self.events.append('original.enter')
            try:
                yield self.yielded_state
            finally:
                self.events.append('original.exit')

        self.app = FastAPI(lifespan=original_lifespan)
        self.app.state.arc3_persistent_manager = self.manager
        self.app.state.arc3_persistent_receipt = {'fake_backend': True}
        PersistentEndpoint().attach_router(self.app)

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(
            app=self.app, raise_app_exceptions=False), base_url='http://testserver')

    async def test_health_start_append_status_close_and_retry(self):
        async with self.app.router.lifespan_context(self.app) as yielded:
            self.assertEqual(yielded, self.yielded_state)
            async with self.client() as client:
                health = await client.get('/arc3/persistent/health')
                self.assertEqual(health.status_code, 200, health.text)
                self.assertFalse(health.json()['prefix_caching'])
                self.assertEqual(health.json()['max_sessions'], 2)
                body = start_body()
                first = await client.post('/arc3/persistent/start', json=body)
                self.assertEqual(first.status_code, 200, first.text)
                receipt = first.json()['persistent']
                self.assertEqual(receipt['revision'], 1)
                self.assertEqual(receipt['resident_tokens'], 3)
                self.assertEqual(receipt['prefix_sha256'], token_digest([1,2,101]))
                repeated = await client.post('/arc3/persistent/start', json=body)
                self.assertEqual(repeated.json(), first.json())
                self.assertEqual(len(self.backend.submitted), 1)
                changed = await client.post('/arc3/persistent/start', json=start_body(ids=[2,1]))
                self.assertEqual(changed.status_code, 409, changed.text)
                append = dict(request_id='append', expected_revision=1,
                    expected_prefix_sha256=receipt['prefix_sha256'], raw={'token_ids':[3]})
                second = await client.post('/arc3/persistent/game/append', json=append)
                self.assertEqual(second.status_code, 200, second.text)
                self.assertEqual(second.json()['usage'], dict(prompt_tokens=4, completion_tokens=1, total_tokens=5))
                again = await client.post('/arc3/persistent/game/append', json=append)
                self.assertEqual(again.json(), second.json())
                self.assertEqual(self.backend.submitted, [('game',[1,2]), ('game',[3])])
                status = await client.get('/arc3/persistent/game/status')
                self.assertEqual(status.json()['state'], 'paused')
                self.assertEqual(status.json()['operations']['append']['state'], 'complete')
                for _ in range(2):
                    closed = await client.post('/arc3/persistent/game/close', json={})
                    self.assertEqual(closed.status_code, 200, closed.text)
                self.assertFalse(self.backend.owned)
                refused = await client.post('/arc3/persistent/game/append',
                    json=dict(request_id='after-close', expected_revision=2, raw={'token_ids':[4]}))
                self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(self.manager.close_all_count, 1)
        self.assertEqual(self.events[-2:], ['manager.close_all','original.exit'])

    async def test_validation_does_not_submit_or_mutate(self):
        async with self.app.router.lifespan_context(self.app):
            async with self.client() as client:
                bad_id = await client.post('/arc3/persistent/start', json=start_body(sid='bad id'))
                self.assertEqual(bad_id.status_code, 400, bad_id.text)
                unknown = await client.get('/arc3/persistent/unknown/status')
                self.assertEqual(unknown.status_code, 404, unknown.text)
                missing = await client.post('/arc3/persistent/start', json={'session_id':'game'})
                self.assertEqual(missing.status_code, 400, missing.text)
                first = await client.post('/arc3/persistent/start', json=start_body())
                self.assertEqual(first.status_code, 200, first.text)
                original = list(self.manager.get('game').token_ids)
                cases = [
                    (dict(expected_revision=0, raw={'token_ids':[4]}), 409),
                    (dict(expected_revision=1, expected_prefix_sha256='bad', raw={'token_ids':[4]}), 409),
                    (dict(expected_revision=1, raw={'token_ids':[]}), 400),
                    (dict(expected_revision=1, raw={'token_ids':[True]}), 400),
                    (dict(expected_revision=1, raw={'token_ids':[4]}, continuation=True), 400),
                ]
                for i, (case, expected_status) in enumerate(cases):
                    response = await client.post('/arc3/persistent/game/append',
                        json=dict(request_id=f'invalid{i}', **case))
                    self.assertEqual(response.status_code, expected_status, response.text)
                    self.assertEqual(self.manager.get('game').token_ids, original)
                self.assertEqual(len(self.backend.submitted), 1)
                self.assertEqual(self.manager.status('game')['state'], 'paused')
        self.assertEqual(self.manager.close_all_count, 1)
        self.assertFalse(self.backend.owned)

    async def test_capacity_admission_and_empty_continuation(self):
        async with self.app.router.lifespan_context(self.app):
            async with self.client() as client:
                for sid in ('a','b'):
                    response = await client.post('/arc3/persistent/start', json=start_body(sid))
                    self.assertEqual(response.status_code, 200, response.text)
                denied = await client.post('/arc3/persistent/start', json=start_body('c'))
                self.assertEqual(denied.status_code, 429, denied.text)
                continued = await client.post('/arc3/persistent/a/append', json=dict(
                    request_id='continue', expected_revision=1, raw={'token_ids':[]}, continuation=True))
                self.assertEqual(continued.status_code, 200, continued.text)
                self.assertEqual(continued.json()['persistent']['appended_prompt_tokens'], 0)
                self.assertEqual(self.manager.get('a').token_ids, [1,2,101,101])
                overflow = await client.post('/arc3/persistent/a/append', json=dict(
                    request_id='overflow', expected_revision=2, raw={'token_ids':list(range(16))}))
                self.assertEqual(overflow.status_code, 413, overflow.text)
                self.assertEqual(self.manager.get('a').token_ids, [1,2,101,101])
                await client.post('/arc3/persistent/a/close', json={})
                admitted = await client.post('/arc3/persistent/start', json=start_body('c'))
                self.assertEqual(admitted.status_code, 200, admitted.text)
        self.assertEqual(self.manager.close_all_count, 1)
        self.assertFalse(self.backend.owned)

    async def test_lifespan_releases_paused_sessions_before_original_exit(self):
        async with self.app.router.lifespan_context(self.app) as yielded:
            self.assertEqual(yielded, self.yielded_state)
            async with self.client() as client:
                response = await client.post('/arc3/persistent/start', json=start_body())
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(self.backend.owned, {'game'})
        self.assertEqual(self.manager.close_all_count, 1)
        self.assertFalse(self.backend.owned)
        self.assertEqual(self.manager.get('game').state, 'closed')
        self.assertLess(self.events.index('manager.close_all'), self.events.index('original.exit'))
        self.assertLess(self.events.index('backend.close:game'), self.events.index('original.exit'))

    async def test_lifespan_body_exception_still_releases_once(self):
        with self.assertRaisesRegex(RuntimeError, 'deliberate lifespan body failure'):
            async with self.app.router.lifespan_context(self.app):
                async with self.client() as client:
                    response = await client.post('/arc3/persistent/start', json=start_body())
                    self.assertEqual(response.status_code, 200, response.text)
                raise RuntimeError('deliberate lifespan body failure')
        self.assertEqual(self.manager.close_all_count, 1)
        self.assertEqual(self.events.count('original.enter'), 1)
        self.assertEqual(self.events.count('original.exit'), 1)
        self.assertFalse(self.backend.owned)

    async def test_lifespan_original_exit_exception_cleans_first(self):
        @asynccontextmanager
        async def failing_exit(app):
            try:
                yield {'exit_exception_fixture': True}
            finally:
                self.events.append('failing.exit')
                raise RuntimeError('deliberate original exit failure')
        app = FastAPI(lifespan=failing_exit)
        app.state.arc3_persistent_manager = self.manager
        PersistentEndpoint().attach_router(app)
        with self.assertRaisesRegex(RuntimeError, 'deliberate original exit failure'):
            async with app.router.lifespan_context(app):
                await self.manager.start(start_body())
        self.assertEqual(self.manager.close_all_count, 1)
        self.assertFalse(self.backend.owned)
        self.assertLess(self.events.index('backend.close:game'), self.events.index('failing.exit'))

    async def test_lifespan_body_exception_cancels_inflight_route(self):
        self.backend.gate = asyncio.Event()
        caller = None
        client = self.client()
        try:
            with self.assertRaisesRegex(RuntimeError, 'inflight body failure'):
                async with self.app.router.lifespan_context(self.app):
                    caller = asyncio.create_task(client.post('/arc3/persistent/start', json=start_body()))
                    async def wait_for_submission():
                        while not self.backend.submitted:
                            await asyncio.sleep(0)
                    await asyncio.wait_for(wait_for_submission(), timeout=1)
                    self.assertEqual(self.backend.owned, {'game'})
                    raise RuntimeError('inflight body failure')
            pending_result = await asyncio.gather(caller, return_exceptions=True)
            self.assertIsInstance(pending_result[0], asyncio.CancelledError)
            self.assertEqual(self.manager.close_all_count, 1)
            self.assertFalse(self.backend.owned)
            self.assertEqual(self.manager.get('game').state, 'closed')
        finally:
            if caller is not None and not caller.done():
                caller.cancel()
                await asyncio.gather(caller, return_exceptions=True)
            await client.aclose()

    async def test_lifespan_without_initialized_manager_preserves_original(self):
        del self.app.state.arc3_persistent_manager
        async with self.app.router.lifespan_context(self.app) as yielded:
            self.assertEqual(yielded, self.yielded_state)
        self.assertEqual(self.events, ['original.enter','original.exit'])
        self.assertEqual(self.manager.close_all_count, 0)

    async def test_failed_engine_request_reports_500_and_frees_ownership(self):
        self.backend.fail_after_submit = True
        async with self.app.router.lifespan_context(self.app):
            async with self.client() as client:
                body = start_body()
                for _ in range(2):
                    response = await client.post('/arc3/persistent/start', json=body)
                    self.assertEqual(response.status_code, 500, response.text)
                    self.assertIn('Deliberate fake engine failure', response.text)
                self.assertEqual(len(self.backend.submitted), 1)
                self.assertFalse(self.backend.owned)
                status = await client.get('/arc3/persistent/game/status')
                self.assertEqual(status.json()['state'], 'error')
        self.assertEqual(self.manager.close_all_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Optional JSON result path')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeRoutes)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    here = Path(__file__).resolve().parent
    report = dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=[{'test':str(test), 'traceback':text} for test,text in result.failures],
        errors=[{'test':str(test), 'traceback':text} for test,text in result.errors],
        scope='Actual FastAPI/httpx ASGI routes with fake backend; no network or GPU use',
        versions={package:importlib.metadata.version(package) for package in ('fastapi','starlette','httpx')},
        python=sys.version,
        source_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
                       for name in ('runtime_route_checks.py','persistent_endpoint.py','session_service.py')},
        untested_note='Original lifespan entry failure before yield needs a separately guarded cleanup path.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('passed','tests','versions','scope')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
