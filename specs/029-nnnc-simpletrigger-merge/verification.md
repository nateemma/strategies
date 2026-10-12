# Verification: 029 NNNC / SimpleTrigger merge (T021–T026)

Code: `5ef1b7d` (the refactor) vs the baseline `6c978e8`. Pass condition: every after-change run (`_a029`) reproduces its same-day pre-change baseline (`_b029`); the record is shown alongside.

## Summary

| check | result |
|---|---|
| `wf_aro` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`303456c1fa322b4c`) |
| `wf_trgate` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`303456c1fa322b4c`) |
| `st_gbb` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`303456c1fa322b4c`) |
| `st_bgo_mx` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`95dd915b90fc6217`) |
| `st_book_rev2` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`0e552be31732d222`) |
| `st_reg_volreg2` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`f1492b64a9dd9257`) |
| `st_coex_win4` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`813a8f67bdb5d847`) |
| `st_coex_cap3` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`bd291047db09d7ab`) |
| `st_pre_volreg2` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`9e8fa9c67a56647e`) |
| `st_prom_c2` after = baseline (record) | **15/15** (15/15); framework_sha unchanged (`2a1bb9f427343c56`) |
| twin (`wf_trgate_a029`) = `st_gbb_a029` | **15/15** |
| NNNC strategy fingerprints (`scripts/nnnc_fingerprints.py --compare`) | 297 strategies: 0 changed, 0 removed, 0 added; 2 pre-existing load errors unchanged; new non-strategy file `BorrowedModelMixin` |
| pinned shas (`evaluation/tests/test_fingerprint_switches.py`, 10 classes) | pass |
| model artefacts `saved_data/NNNC_ARO_f*/` (sha256, 15 files) | unchanged |
| training (SC-005) | 0 × `training on`; 30 × `model already present` (arm D + twin) |
| test suites | 885 passed; 3 failed = the pre-existing training-pipeline goldens (`Framework/tests/test_training_pipeline_characterization.py`, fail identically before 029) |
| stop invariants on every NNNC arm (T026) | every NNNC strategy's MRO carries `StopInvariantMixin` once, before `BaseStrategy` (`NNNC/tests/test_nnnc_merge.py`); arm D's after-change run carried the HALTING check and completed 15/15; a planted violation halts an NNNC instance (T027) |

Record-level differences that do not touch trades (both expected):
- arm D's entries are tagged `reversion:model` (A′: `gbb_path` → reversion) instead of `model_entry `; `trades_hash` excludes tags and no NNNC class declares family exits.
- the twin's record field `trigger` is now `null`: its trigger moved from `TRIGGER_NAME` to `TWIN_TRIGGER` (a declared trigger on a model-mode strategy is refused as ambiguous).

## Per fold

| arm | fold | baseline | after | record |
|---|---|---|---|---|
| `wf_aro` | f00 | 5c53c833e423eae4 | 5c53c833e423eae4 | 5c53c833e423eae4 |
| `wf_aro` | f01 | ba30d7ee37144509 | ba30d7ee37144509 | ba30d7ee37144509 |
| `wf_aro` | f02 | d30b4c4d8424b9bb | d30b4c4d8424b9bb | d30b4c4d8424b9bb |
| `wf_aro` | f03 | 0fd8e37cf8cb2608 | 0fd8e37cf8cb2608 | 0fd8e37cf8cb2608 |
| `wf_aro` | f04 | 589a8a98faef6389 | 589a8a98faef6389 | 589a8a98faef6389 |
| `wf_aro` | f05 | d35467f2f7f51e94 | d35467f2f7f51e94 | d35467f2f7f51e94 |
| `wf_aro` | f06 | d41b99b494482847 | d41b99b494482847 | d41b99b494482847 |
| `wf_aro` | f07 | 13d84f2c7a3be2b9 | 13d84f2c7a3be2b9 | 13d84f2c7a3be2b9 |
| `wf_aro` | f08 | aae5a44310f5e173 | aae5a44310f5e173 | aae5a44310f5e173 |
| `wf_aro` | f09 | 6e8b99b5da07525d | 6e8b99b5da07525d | 6e8b99b5da07525d |
| `wf_aro` | f10 | 5d4ac5826f7f800a | 5d4ac5826f7f800a | 5d4ac5826f7f800a |
| `wf_aro` | f11 | 54d3293558d51208 | 54d3293558d51208 | 54d3293558d51208 |
| `wf_aro` | f12 | 6918556795bf97fe | 6918556795bf97fe | 6918556795bf97fe |
| `wf_aro` | f13 | 4ef2a6e041ab2d96 | 4ef2a6e041ab2d96 | 4ef2a6e041ab2d96 |
| `wf_aro` | f14 | c3a819f6b1316ba5 | c3a819f6b1316ba5 | c3a819f6b1316ba5 |
| `wf_trgate` | f00 | aa058ba815f48627 | aa058ba815f48627 | aa058ba815f48627 |
| `wf_trgate` | f01 | 55166215f9ac31aa | 55166215f9ac31aa | 55166215f9ac31aa |
| `wf_trgate` | f02 | 1f690367a7690e40 | 1f690367a7690e40 | 1f690367a7690e40 |
| `wf_trgate` | f03 | b4d7d37b1d204f1a | b4d7d37b1d204f1a | b4d7d37b1d204f1a |
| `wf_trgate` | f04 | b16ebf1dd6505067 | b16ebf1dd6505067 | b16ebf1dd6505067 |
| `wf_trgate` | f05 | 9419a8736467df24 | 9419a8736467df24 | 9419a8736467df24 |
| `wf_trgate` | f06 | 28254aea3c90a257 | 28254aea3c90a257 | 28254aea3c90a257 |
| `wf_trgate` | f07 | 6b1a542b6682bf2f | 6b1a542b6682bf2f | 6b1a542b6682bf2f |
| `wf_trgate` | f08 | 758b4d2cc4e314cf | 758b4d2cc4e314cf | 758b4d2cc4e314cf |
| `wf_trgate` | f09 | 09e8027c786ba9d5 | 09e8027c786ba9d5 | 09e8027c786ba9d5 |
| `wf_trgate` | f10 | 7c2d55c184801fcc | 7c2d55c184801fcc | 7c2d55c184801fcc |
| `wf_trgate` | f11 | e3053d309ffa5bc5 | e3053d309ffa5bc5 | e3053d309ffa5bc5 |
| `wf_trgate` | f12 | 3dc89f0a8724f854 | 3dc89f0a8724f854 | 3dc89f0a8724f854 |
| `wf_trgate` | f13 | e9d0bf579c58755f | e9d0bf579c58755f | e9d0bf579c58755f |
| `wf_trgate` | f14 | 8b4d0c18696801fc | 8b4d0c18696801fc | 8b4d0c18696801fc |
| `st_gbb` | f00 | aa058ba815f48627 | aa058ba815f48627 | aa058ba815f48627 |
| `st_gbb` | f01 | 55166215f9ac31aa | 55166215f9ac31aa | 55166215f9ac31aa |
| `st_gbb` | f02 | 1f690367a7690e40 | 1f690367a7690e40 | 1f690367a7690e40 |
| `st_gbb` | f03 | b4d7d37b1d204f1a | b4d7d37b1d204f1a | b4d7d37b1d204f1a |
| `st_gbb` | f04 | b16ebf1dd6505067 | b16ebf1dd6505067 | b16ebf1dd6505067 |
| `st_gbb` | f05 | 9419a8736467df24 | 9419a8736467df24 | 9419a8736467df24 |
| `st_gbb` | f06 | 28254aea3c90a257 | 28254aea3c90a257 | 28254aea3c90a257 |
| `st_gbb` | f07 | 6b1a542b6682bf2f | 6b1a542b6682bf2f | 6b1a542b6682bf2f |
| `st_gbb` | f08 | 758b4d2cc4e314cf | 758b4d2cc4e314cf | 758b4d2cc4e314cf |
| `st_gbb` | f09 | 09e8027c786ba9d5 | 09e8027c786ba9d5 | 09e8027c786ba9d5 |
| `st_gbb` | f10 | 7c2d55c184801fcc | 7c2d55c184801fcc | 7c2d55c184801fcc |
| `st_gbb` | f11 | e3053d309ffa5bc5 | e3053d309ffa5bc5 | e3053d309ffa5bc5 |
| `st_gbb` | f12 | 3dc89f0a8724f854 | 3dc89f0a8724f854 | 3dc89f0a8724f854 |
| `st_gbb` | f13 | e9d0bf579c58755f | e9d0bf579c58755f | e9d0bf579c58755f |
| `st_gbb` | f14 | 8b4d0c18696801fc | 8b4d0c18696801fc | 8b4d0c18696801fc |
| `st_bgo_mx` | f00 | 010350f094a0c19e | 010350f094a0c19e | 010350f094a0c19e |
| `st_bgo_mx` | f01 | 166b3f0ab21d4c70 | 166b3f0ab21d4c70 | 166b3f0ab21d4c70 |
| `st_bgo_mx` | f02 | 0ef46f15159a3827 | 0ef46f15159a3827 | 0ef46f15159a3827 |
| `st_bgo_mx` | f03 | f4c1fed61d2dac4a | f4c1fed61d2dac4a | f4c1fed61d2dac4a |
| `st_bgo_mx` | f04 | 87ac89707f99ac67 | 87ac89707f99ac67 | 87ac89707f99ac67 |
| `st_bgo_mx` | f05 | a598024b1e2f51e2 | a598024b1e2f51e2 | a598024b1e2f51e2 |
| `st_bgo_mx` | f06 | 995105814fba6b56 | 995105814fba6b56 | 995105814fba6b56 |
| `st_bgo_mx` | f07 | 2d362bc9aad94118 | 2d362bc9aad94118 | 2d362bc9aad94118 |
| `st_bgo_mx` | f08 | 690f671a476915e1 | 690f671a476915e1 | 690f671a476915e1 |
| `st_bgo_mx` | f09 | fe5d363c38d15a99 | fe5d363c38d15a99 | fe5d363c38d15a99 |
| `st_bgo_mx` | f10 | 5b476d708128a73c | 5b476d708128a73c | 5b476d708128a73c |
| `st_bgo_mx` | f11 | caeccb48b8bd96f6 | caeccb48b8bd96f6 | caeccb48b8bd96f6 |
| `st_bgo_mx` | f12 | 040f3a51b7c426af | 040f3a51b7c426af | 040f3a51b7c426af |
| `st_bgo_mx` | f13 | 0923a8f75b883640 | 0923a8f75b883640 | 0923a8f75b883640 |
| `st_bgo_mx` | f14 | ec67b3f61fc22390 | ec67b3f61fc22390 | ec67b3f61fc22390 |
| `st_book_rev2` | f00 | 26869046bc547532 | 26869046bc547532 | 26869046bc547532 |
| `st_book_rev2` | f01 | 1258ca8f31ff5410 | 1258ca8f31ff5410 | 1258ca8f31ff5410 |
| `st_book_rev2` | f02 | 388a52682d564b6a | 388a52682d564b6a | 388a52682d564b6a |
| `st_book_rev2` | f03 | 51b4c195e25adc93 | 51b4c195e25adc93 | 51b4c195e25adc93 |
| `st_book_rev2` | f04 | dc4c150b919227d2 | dc4c150b919227d2 | dc4c150b919227d2 |
| `st_book_rev2` | f05 | 2a988ad14da4f912 | 2a988ad14da4f912 | 2a988ad14da4f912 |
| `st_book_rev2` | f06 | 9b4c5ffe469d1544 | 9b4c5ffe469d1544 | 9b4c5ffe469d1544 |
| `st_book_rev2` | f07 | fb5cbd1d2a05662e | fb5cbd1d2a05662e | fb5cbd1d2a05662e |
| `st_book_rev2` | f08 | 9ea7274079158641 | 9ea7274079158641 | 9ea7274079158641 |
| `st_book_rev2` | f09 | 1214af2bcba5ccfc | 1214af2bcba5ccfc | 1214af2bcba5ccfc |
| `st_book_rev2` | f10 | eea79a90d928285f | eea79a90d928285f | eea79a90d928285f |
| `st_book_rev2` | f11 | 4468aefce940827f | 4468aefce940827f | 4468aefce940827f |
| `st_book_rev2` | f12 | 0dc9944276a3d7ae | 0dc9944276a3d7ae | 0dc9944276a3d7ae |
| `st_book_rev2` | f13 | f31689c551ef1d95 | f31689c551ef1d95 | f31689c551ef1d95 |
| `st_book_rev2` | f14 | 16afd9f84e4a139d | 16afd9f84e4a139d | 16afd9f84e4a139d |
| `st_reg_volreg2` | f00 | 9490d57f57a374aa | 9490d57f57a374aa | 9490d57f57a374aa |
| `st_reg_volreg2` | f01 | 0d9ace86535b6573 | 0d9ace86535b6573 | 0d9ace86535b6573 |
| `st_reg_volreg2` | f02 | eebabffcdf8eac81 | eebabffcdf8eac81 | eebabffcdf8eac81 |
| `st_reg_volreg2` | f03 | 86f90249b03a08cb | 86f90249b03a08cb | 86f90249b03a08cb |
| `st_reg_volreg2` | f04 | 4d642dbbae983078 | 4d642dbbae983078 | 4d642dbbae983078 |
| `st_reg_volreg2` | f05 | e828bf8ed7a29879 | e828bf8ed7a29879 | e828bf8ed7a29879 |
| `st_reg_volreg2` | f06 | 3d2e2f81caf494bd | 3d2e2f81caf494bd | 3d2e2f81caf494bd |
| `st_reg_volreg2` | f07 | 7e0ba2fa31f75f99 | 7e0ba2fa31f75f99 | 7e0ba2fa31f75f99 |
| `st_reg_volreg2` | f08 | 5ed2b7a6cfc1e248 | 5ed2b7a6cfc1e248 | 5ed2b7a6cfc1e248 |
| `st_reg_volreg2` | f09 | 5aad6a0bdcec85cf | 5aad6a0bdcec85cf | 5aad6a0bdcec85cf |
| `st_reg_volreg2` | f10 | 2c75aada0e790abb | 2c75aada0e790abb | 2c75aada0e790abb |
| `st_reg_volreg2` | f11 | 86b6d6629cdadc6f | 86b6d6629cdadc6f | 86b6d6629cdadc6f |
| `st_reg_volreg2` | f12 | fb2e2c6c429f3602 | fb2e2c6c429f3602 | fb2e2c6c429f3602 |
| `st_reg_volreg2` | f13 | a26403176ffacd73 | a26403176ffacd73 | a26403176ffacd73 |
| `st_reg_volreg2` | f14 | 21fac282e245a6a8 | 21fac282e245a6a8 | 21fac282e245a6a8 |
| `st_coex_win4` | f00 | 1392ce0a7d2e48f6 | 1392ce0a7d2e48f6 | 1392ce0a7d2e48f6 |
| `st_coex_win4` | f01 | 212f3f52ebaaa3d4 | 212f3f52ebaaa3d4 | 212f3f52ebaaa3d4 |
| `st_coex_win4` | f02 | 0161281109f84cd7 | 0161281109f84cd7 | 0161281109f84cd7 |
| `st_coex_win4` | f03 | 3c0ae763a4d2c9ab | 3c0ae763a4d2c9ab | 3c0ae763a4d2c9ab |
| `st_coex_win4` | f04 | 0df0ff4599aa4d2e | 0df0ff4599aa4d2e | 0df0ff4599aa4d2e |
| `st_coex_win4` | f05 | aef28b83b316f803 | aef28b83b316f803 | aef28b83b316f803 |
| `st_coex_win4` | f06 | 12c97af0a4771759 | 12c97af0a4771759 | 12c97af0a4771759 |
| `st_coex_win4` | f07 | 0de357ba3c55e5de | 0de357ba3c55e5de | 0de357ba3c55e5de |
| `st_coex_win4` | f08 | 060f8ca18ca061e8 | 060f8ca18ca061e8 | 060f8ca18ca061e8 |
| `st_coex_win4` | f09 | 859fc07a68344464 | 859fc07a68344464 | 859fc07a68344464 |
| `st_coex_win4` | f10 | 2989fae93455da47 | 2989fae93455da47 | 2989fae93455da47 |
| `st_coex_win4` | f11 | f0a6da2d29b93eba | f0a6da2d29b93eba | f0a6da2d29b93eba |
| `st_coex_win4` | f12 | 66a757422a3e8f48 | 66a757422a3e8f48 | 66a757422a3e8f48 |
| `st_coex_win4` | f13 | 11403ba64a30d093 | 11403ba64a30d093 | 11403ba64a30d093 |
| `st_coex_win4` | f14 | 8121260bc6f04588 | 8121260bc6f04588 | 8121260bc6f04588 |
| `st_coex_cap3` | f00 | 9463e003e98a0881 | 9463e003e98a0881 | 9463e003e98a0881 |
| `st_coex_cap3` | f01 | 245371502afc63eb | 245371502afc63eb | 245371502afc63eb |
| `st_coex_cap3` | f02 | 552f35aa4b0b828f | 552f35aa4b0b828f | 552f35aa4b0b828f |
| `st_coex_cap3` | f03 | 6387086c3a0334dc | 6387086c3a0334dc | 6387086c3a0334dc |
| `st_coex_cap3` | f04 | 022b6bf0e0b474e1 | 022b6bf0e0b474e1 | 022b6bf0e0b474e1 |
| `st_coex_cap3` | f05 | 68a408a1aa32a477 | 68a408a1aa32a477 | 68a408a1aa32a477 |
| `st_coex_cap3` | f06 | 3a1f31bfa5e30e97 | 3a1f31bfa5e30e97 | 3a1f31bfa5e30e97 |
| `st_coex_cap3` | f07 | 3ab9eaabb2a31f24 | 3ab9eaabb2a31f24 | 3ab9eaabb2a31f24 |
| `st_coex_cap3` | f08 | 1d729bda5afcdfd3 | 1d729bda5afcdfd3 | 1d729bda5afcdfd3 |
| `st_coex_cap3` | f09 | d6be94407777d98d | d6be94407777d98d | d6be94407777d98d |
| `st_coex_cap3` | f10 | f470cfff8e2ca33c | f470cfff8e2ca33c | f470cfff8e2ca33c |
| `st_coex_cap3` | f11 | dc5969055d1d9f1f | dc5969055d1d9f1f | dc5969055d1d9f1f |
| `st_coex_cap3` | f12 | 78a2ef0b09731a29 | 78a2ef0b09731a29 | 78a2ef0b09731a29 |
| `st_coex_cap3` | f13 | 9f7e5b1001165dbf | 9f7e5b1001165dbf | 9f7e5b1001165dbf |
| `st_coex_cap3` | f14 | 00e7c7c21455fa59 | 00e7c7c21455fa59 | 00e7c7c21455fa59 |
| `st_pre_volreg2` | f00 | 902e43c9516c3659 | 902e43c9516c3659 | 902e43c9516c3659 |
| `st_pre_volreg2` | f01 | 8d3149b25f30406a | 8d3149b25f30406a | 8d3149b25f30406a |
| `st_pre_volreg2` | f02 | 63c23dae4b62fb5f | 63c23dae4b62fb5f | 63c23dae4b62fb5f |
| `st_pre_volreg2` | f03 | d0d1d26db6c650b5 | d0d1d26db6c650b5 | d0d1d26db6c650b5 |
| `st_pre_volreg2` | f04 | 17c73fa7bfaf2d09 | 17c73fa7bfaf2d09 | 17c73fa7bfaf2d09 |
| `st_pre_volreg2` | f05 | 08983ff76692cb7c | 08983ff76692cb7c | 08983ff76692cb7c |
| `st_pre_volreg2` | f06 | 5218a35518e68cca | 5218a35518e68cca | 5218a35518e68cca |
| `st_pre_volreg2` | f07 | 9ae69e9a581fd5b7 | 9ae69e9a581fd5b7 | 9ae69e9a581fd5b7 |
| `st_pre_volreg2` | f08 | c540d20ab502db4a | c540d20ab502db4a | c540d20ab502db4a |
| `st_pre_volreg2` | f09 | 4c6dd09dfa020992 | 4c6dd09dfa020992 | 4c6dd09dfa020992 |
| `st_pre_volreg2` | f10 | 1b3a36073a378f26 | 1b3a36073a378f26 | 1b3a36073a378f26 |
| `st_pre_volreg2` | f11 | 9dee3ec7363a56ad | 9dee3ec7363a56ad | 9dee3ec7363a56ad |
| `st_pre_volreg2` | f12 | e75a67d46199f017 | e75a67d46199f017 | e75a67d46199f017 |
| `st_pre_volreg2` | f13 | 3a06d4b2c822e52d | 3a06d4b2c822e52d | 3a06d4b2c822e52d |
| `st_pre_volreg2` | f14 | 3779deb371be097e | 3779deb371be097e | 3779deb371be097e |
| `st_prom_c2` | f00 | 657bfcf2edb6ef6a | 657bfcf2edb6ef6a | 657bfcf2edb6ef6a |
| `st_prom_c2` | f01 | 173d59c478e9fd7b | 173d59c478e9fd7b | 173d59c478e9fd7b |
| `st_prom_c2` | f02 | 009f70970e59e871 | 009f70970e59e871 | 009f70970e59e871 |
| `st_prom_c2` | f03 | d51d47d2bedea17a | d51d47d2bedea17a | d51d47d2bedea17a |
| `st_prom_c2` | f04 | bea779ffcf2adac9 | bea779ffcf2adac9 | bea779ffcf2adac9 |
| `st_prom_c2` | f05 | b9118e4a374916c3 | b9118e4a374916c3 | b9118e4a374916c3 |
| `st_prom_c2` | f06 | 694572c018a5c2ff | 694572c018a5c2ff | 694572c018a5c2ff |
| `st_prom_c2` | f07 | 151b507ded27a5a5 | 151b507ded27a5a5 | 151b507ded27a5a5 |
| `st_prom_c2` | f08 | 59af3bcad75ce936 | 59af3bcad75ce936 | 59af3bcad75ce936 |
| `st_prom_c2` | f09 | ae6666a04bbb96fd | ae6666a04bbb96fd | ae6666a04bbb96fd |
| `st_prom_c2` | f10 | 95282e4c277af891 | 95282e4c277af891 | 95282e4c277af891 |
| `st_prom_c2` | f11 | 42cee54507d4054c | 42cee54507d4054c | 42cee54507d4054c |
| `st_prom_c2` | f12 | 82d08fa247714982 | 82d08fa247714982 | 82d08fa247714982 |
| `st_prom_c2` | f13 | ccd9ef624b742469 | ccd9ef624b742469 | ccd9ef624b742469 |
| `st_prom_c2` | f14 | a50ca94dbe754a84 | a50ca94dbe754a84 | a50ca94dbe754a84 |
