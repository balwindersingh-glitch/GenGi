# GenGi

This is the GenGi project. Work in progress.

## Deploying to Vercel

- Set **Environment Variables**: `GOOGLE_APPLICATION_CREDENTIALS_JSON` (paste full contents of your GCP service account JSON). Optionally: `VEO_OUTPUT_GCS_URI`, `GOOGLE_CLOUD_PROJECT`, `NANO_BANANA_API_KEY`, `DRIVE_FOLDER_ID`.
- **Function timeout**: In Vercel → Project → **Settings** → **Functions**, set **Default Max Duration** to **120** (or 60) so image/video generation don’t time out.

## Project Files
- analyze_video.py
- app.py
- drive_upload.py
- frontend
- generate_image.py
- generate_video.py
- nano_banana.py
- requirements.txt
- run.sh
- setup-gcs-permissions.sh
\n## Live Site\n- [GenGi Live](https://gen-gi.vercel.app/)
