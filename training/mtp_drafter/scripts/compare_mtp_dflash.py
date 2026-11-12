"""compare_mtp_dflash.py (2026-10-12, Claude fork 5d20b23b): the MTP-vs-DFlash table on val50 from the side pod's result
files: MTP (mtp_eval_val50.py json: prob_srv = production's probabilistic drafts; greedy_srv) and DFlash (dflash_eval_window.py
json: sel_srv = the served greedy-selector path; prob_srv), per T, with throughput normalised by the step time relative to
MTP3 and the parity bars implied by MTP3.
Step costs (relative to the MTP3 step): MTP +~9% per extra draft (measured at K 4, 64K tier: +13% tau, +3.9% per lane;
K 5 extrapolated), DFlash ~1.10x at K 6 and ~1.17x at K 8 (estimates: one verify row ~3.4% per slot). Lanes: shown as the
parity variants (DFlash K 8 ~1 lane short of MTP3 by the 10-11 memory estimates; -1.6 / -4 lanes bars kept from before).
usage: python3 compare_mtp_dflash.py MTP_JSON DFLASH_JSON [--window 512]"""
import json
import sys

MTP_COST = {3: 1.00, 4: 1.09, 5: 1.18}
DFL_COST = {6: 1.10, 8: 1.17}


def main(mtp_path, dfl_path, window='512'):
    m = json.load(open(mtp_path))['summary']
    d = json.load(open(dfl_path))['results'][window]
    for T in ('10', '06'):
        mp, mg = m['prob_srv_T' + T], m['greedy_srv_T' + T]
        print('== T %s.%s  (MTP anchors %d)' % (T[0], T[1], mp['anchors']))
        print('MTP prob_srv  slots %s  L1-5 %s  cond %s' % (mp['slots'], mp['L'], mp['cond']))
        print('MTP greedy    L1-5 %s' % mg['L'])
        sel = [d.get('xa_sel_srv_T%s_L%d' % (T, k)) for k in range(1, 9)]
        prb = [d.get('xa_prob_srv_T%s_L%d' % (T, k)) for k in range(1, 9)]
        slots = [d.get('xa_sel_srv_T%s_s%d' % (T, k)) for k in range(1, 9)]
        print('DFlash sel_srv slots %s' % slots)
        print('DFlash sel_srv L1-8 %s' % sel)
        print('DFlash prob_srv L1-8 %s' % prb)
        tau3 = mp['L'][2]
        print('normalised (tau / step cost vs MTP3):')
        for K, c in MTP_COST.items():
            if K <= len(mp['L']):
                print('  MTP K %d: %.3f / %.2f = %.3f' % (K, mp['L'][K - 1], c, mp['L'][K - 1] / c))
        for K, c in DFL_COST.items():
            if sel[K - 1] is not None:
                print('  DFlash K %d: %.3f / %.2f = %.3f (%.1f%% of MTP3)' % (K, sel[K - 1], c, sel[K - 1] / c, 100 * sel[K - 1] / c / tau3))
        print('parity bars from MTP3 tau %.3f: K 6 L6 >= %.3f; K 8 L8 >= %.3f (no lane loss), %.3f (-1.6 lanes), %.3f (-4 lanes)' % (
            tau3, tau3 * 1.10, tau3 * 1.17, tau3 * 1.17 * 32 / 30.4, tau3 * 1.17 * 32 / 28))
        best = max(mp['L'][K - 1] / c for K, c in MTP_COST.items() if K <= len(mp['L']))
        print('vs MTP at its best K (%.3f): K 6 L6 >= %.3f, K 8 L8 >= %.3f' % (best, best * 1.10, best * 1.17))


if __name__ == '__main__':
    main(*sys.argv[1:3], *(sys.argv[3:4] or ['512']))
