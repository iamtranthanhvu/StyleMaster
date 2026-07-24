# TODO: StyleMaster Image-to-Image

Chi tiết & tiêu chí ở `tasks/plan.md`. Thứ tự tuyến tính A→D (không chạy song song).

## Phase 1 — Foundation (máy dev)
- [ ] **Task 1** — Commit + push code i2i (SPEC, .gitignore, i2i script, run_i2i.sh)
  - Verify: `git show --stat HEAD` đúng file; không có weights; `git push` OK
- [ ] **Checkpoint:** code ở remote, không kèm weights

## Phase 2 — Môi trường host (host GPU)
- [ ] **Task 2** — `git pull` + `pip install -e .` + `python download_ckpt.py`
  - Verify: 2 file `.ckpt`; 3 file Wan2.1; `torch.cuda.is_available()` = True
- [ ] **Checkpoint:** weights + GPU sẵn sàng

## Phase 3 — End-to-end (host GPU) ⚠️ rủi ro cao
- [ ] **Task 3** — Chạy `run_i2i.sh` lần đầu → `results/i2i/out.png`
  - Verify: PNG mở được; log shape `(1,3,1,480,832)`
  - Fallback: `NUM_FRAMES=5 bash run_i2i.sh ...` nếu lỗi temporal=1
- [ ] **Task 4** — Xác nhận output giữ bố cục content + đúng style (thử thêm 1 style khác)
- [ ] **Checkpoint:** i2i end-to-end đúng nghĩa restyle

## Phase 4 — Tinh chỉnh (host GPU)
- [ ] **Task 5** — Dò ≥3 cấu hình `STYLE_CFG`/`CN_SCALE`/`CN_END`, chốt mặc định, ghi vào SPEC
- [ ] **Checkpoint:** hoàn tất, đạt T1–T4 của SPEC

## Cần bạn cung cấp
- [ ] 1 ảnh content để test (repo chưa có) — hoặc trích frame từ `example_test_data/girl.mp4`
- [ ] Quyết định: có commit `CLAUDE.md` không?
