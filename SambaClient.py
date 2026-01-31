import os
import smbclient
from smbclient import register_session, open_file

class SambaClient:
    def __init__(self, server: str, share: str, user: str, password: str):
        self.server = server
        self.share = share
        self.user = user
        self.password = password
        register_session(self.server, username=self.user, password=self.password)

    def _remote(self, path: str) -> str:
        path = path.lstrip("/").replace("/", "\\")
        return fr"\\{self.server}\{self.share}\{path}"

    def listdir(self, dir_path: str):
        return smbclient.listdir(self._remote(dir_path))

    def scandir(self, dir_path: str):
        return smbclient.scandir(self._remote(dir_path))  # yields DirEntry objects

    def mkdir(self, dir_path: str):
        smbclient.mkdir(self._remote(dir_path))

    def makedirs(self, dir_path: str, exist_ok: bool = True):
        smbclient.makedirs(self._remote(dir_path), exist_ok=exist_ok)

    def rmdir(self, dir_path: str):
        smbclient.rmdir(self._remote(dir_path))  # must be empty

    def remove(self, file_path: str):
        smbclient.remove(self._remote(file_path))

    def rename(self, src: str, dst: str):
        smbclient.rename(self._remote(src), self._remote(dst))

    def rmtree(self, dir_path: str):
        """Recursive delete like shutil.rmtree."""
        rp = self._remote(dir_path)
        for entry in smbclient.scandir(rp):
            child = rp.rstrip("\\") + "\\" + entry.name
            if entry.is_dir():
                self.rmtree(child.replace(fr"\\{self.server}\{self.share}\\", ""))
            else:
                smbclient.remove(child)
        smbclient.rmdir(rp)

    def write_file(self, local_path: str, remote_path: str, overwrite: bool = True, chunk_size: int = 1024 * 1024):
        """
        Copy any local file (e.g. mp4, zip, model weights, etc.) to the SMB server.

        local_path: path on your machine (e.g. /home/user/video.mp4)
        remote_path: path inside the SMB share (e.g. videos/video.mp4)
        """
        self.ensure_parent_dir(remote_path)
        mode = "wb" if overwrite else "xb"
        with open(local_path, "rb") as src:
            with open_file(self._remote(remote_path), mode) as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)

    def read_file(self, remote_path: str, local_path: str, chunk_size: int = 1024 * 1024):
        """
        Download a file from the SMB server to your local machine.

        remote_path: path inside the SMB share (e.g. videos/video.mp4)
        local_path: path on your machine (e.g. /home/user/downloads/video.mp4)
        """
        # Ensure the local directory exists before writing
        local_dir = os.path.dirname(local_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        with open_file(self._remote(remote_path), mode="rb") as src:
            with open(local_path, "wb") as dst:
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)

    def ensure_parent_dir(self, path: str):
        """
        Ensure the parent directory of `path` exists on the SMB share.
        """
        parent = os.path.dirname(path)
        if not parent or parent == "/":
            return
        smb_parent = self._remote(parent)
        try:
            smbclient.makedirs(smb_parent, exist_ok=True)
        except FileExistsError:
            pass

    def exists(self, path: str) -> bool:
        """
        Check if a file or directory exists on the SMB share.
        """
        try:
            smbclient.stat(self._remote(path))
            return True
        except (OSError, FileNotFoundError):
            return False