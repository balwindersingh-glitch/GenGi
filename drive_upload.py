"""
Upload a local file to Google Drive using application default credentials (service account).
Set GOOGLE_APPLICATION_CREDENTIALS and optionally DRIVE_FOLDER_ID.
"""
import os
from pathlib import Path
from typing import Optional

SCOPES = ["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]


def upload_to_drive(
    local_path: str,
    name: Optional[str] = None,
    folder_id: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> dict:
    """
    Upload file to Drive. Returns dict with id, webViewLink, webContentLink.
    folder_id from env DRIVE_FOLDER_ID if not passed.
    """
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(local_path)
    folder_id = folder_id or os.environ.get("DRIVE_FOLDER_ID", "")
    name = name or path.name

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.isfile(creds_path):
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set or file missing")
    credentials = service_account.Credentials.from_service_account_file(
        creds_path, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)

    body = {"name": name}
    if folder_id:
        body["parents"] = [folder_id]
    if not mime_type:
        mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "video/mp4"
    media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
    file = service.files().create(body=body, media_body=media, fields="id,webViewLink,webContentLink").execute()

    # Optional: make viewable by anyone with link (comment out if not needed)
    try:
        service.permissions().create(
            fileId=file["id"],
            body={"type": "anyone", "role": "reader"},
        ).execute()
    except Exception:
        pass

    return {
        "id": file.get("id"),
        "webViewLink": file.get("webViewLink", ""),
        "webContentLink": file.get("webContentLink", ""),
    }
