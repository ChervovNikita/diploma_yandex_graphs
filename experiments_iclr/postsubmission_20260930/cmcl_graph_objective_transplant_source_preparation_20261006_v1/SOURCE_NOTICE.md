# CMCL provenance and adaptations

Objective attribution: **Confident Multiple Choice Learning**, ICML2017, PMLR70/lee17b, https://proceedings.mlr.press/v70/lee17b/lee17b.pdf.

The paper authenticates author repository https://github.com/chhwang/cmcl. Saved objective code is from commit **57f41f4b166b8544c56580f9d32ee98b997e3f59**, dated2018-08-04, exact treee15e9160e4627c5e05f87706ffed90a581653a37. The committed LICENSE is Apache License2.0, blobf953bdfc05d8740212d6a879162b276b6e2626ce, SHA256143b5ec92c68fa0d6a02ecc8525ee3564be6e3d511168a0176beca92bb616ceb; its full original text is retained in AUTHOR_LICENSE.txt.

Preserved upstream notice: **Copyright2018, Kimin Lee, Changho Hwang, KyoungSoo Park, and Jinwoo Shin.** The upstream code is provided under Apache2.0 and its AS-IS terms. No claim is made that this2018 snapshot is the exact2017 experiment revision.

cmcl_loss.py is independently written from the attributed objective equations and authenticated ordering/normalization. No TensorFlow source body is vendored or changed. It replaces clipped probability logs with stable exact log_softmax, uses the existing shared native graph bank with M4/K3/beta.75, equal S/R role means and declared plain SGD in the prospective contract, and omits optional stochastic masked feature exchange/stochastic-label version1/image optimizer schedules. These are documented objective-transplant choices, not an image replication or published optimal setting. The author's optional feature exchange is distinct from the existing tied native core, which stays unchanged.

The original K1 Algorithm1 primary scope and every design packet remain preserved. The saved close design supersedes K1 as the recommended sole comparator with fixed K3; this helper exposes no K/beta search or alternate K1 arm. The neighbor-vote proposal is not implemented.
