from typing import IO, Any
import pyfuse3
import pyfuse3_asyncio
from pathlib import Path
import asyncio
import logging
import os
import errno
import stat

pyfuse3_asyncio.enable()
logger = logging.getLogger(__name__)
# based on https://github.com/libfuse/pyfuse3/blob/master/examples/hello_asyncio.py

class MountedClientFile:
    def __init__(self, fobj: IO[Any], name: str):
        self.fuse_options = set(pyfuse3.default_options)
        self.fuse_options.add("fsname=hello_asyncio")
        debug_fuse = True
        if debug_fuse:
            self.fuse_options.add("debug")
        self.fs = pyfuseFilesystem(fobj, name)
        self.mountpoint = Path(f"./fuse3/client/")
        self.path = self.mountpoint / name
        self.mountpoint.mkdir(exist_ok=True, parents=True)

    async def start(self):
        pyfuse3.init(self.fs, str(self.mountpoint), self.fuse_options)
        try:
            self.task = asyncio.create_task(pyfuse3.main())
        except:
            pyfuse3.close()
            raise

    async def close(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass

    def __del__(self):
        pyfuse3.close()


class pyfuseFilesystem(pyfuse3.Operations):
    def __init__(self, fobj: IO[Any], name: str):
        super(pyfuseFilesystem, self).__init__()
        self.hello_name = name.encode()
        self.hello_inode = pyfuse3.ROOT_INODE + 1
        self.fobj = fobj

    async def getattr(self, inode, ctx=None):
        entry = pyfuse3.EntryAttributes()
        if inode == pyfuse3.ROOT_INODE:
            entry.st_mode = stat.S_IFDIR | 0o755
            entry.st_size = 0
        elif inode == self.hello_inode:
            entry.st_mode = stat.S_IFREG | 0o644
            entry.st_size = self.fobj.size
        else:
            raise pyfuse3.FUSEError(errno.ENOENT)

        stamp = int(1438467123.985654 * 1e9)
        entry.st_atime_ns = stamp
        entry.st_ctime_ns = stamp
        entry.st_mtime_ns = stamp
        entry.st_gid = os.getgid()
        entry.st_uid = os.getuid()
        entry.st_ino = inode

        return entry

    async def lookup(self, parent_inode, name, ctx=None):
        if parent_inode != pyfuse3.ROOT_INODE or name != self.hello_name:
            raise pyfuse3.FUSEError(errno.ENOENT)
        return await self.getattr(self.hello_inode)

    async def opendir(self, inode, ctx):
        if inode != pyfuse3.ROOT_INODE:
            raise pyfuse3.FUSEError(errno.ENOENT)
        return inode

    async def readdir(self, fh, start_id, token):
        assert fh == pyfuse3.ROOT_INODE

        # only one entry
        if start_id == 0:
            pyfuse3.readdir_reply(
                token, self.hello_name, await self.getattr(self.hello_inode), 1
            )
        return

    async def setxattr(self, inode, name, value, ctx):
        if inode != pyfuse3.ROOT_INODE or name != b"command":
            raise pyfuse3.FUSEError(errno.ENOTSUP)

        if value == b"terminate":
            pyfuse3.terminate()
        else:
            raise pyfuse3.FUSEError(errno.EINVAL)

    async def open(self, inode, flags, ctx):
        if inode != self.hello_inode:
            raise pyfuse3.FUSEError(errno.ENOENT)
        if flags & os.O_RDWR or flags & os.O_WRONLY:
            raise pyfuse3.FUSEError(errno.EACCES)
        return pyfuse3.FileInfo(fh=inode)

    async def read(self, fh, off, size):
        assert fh == self.hello_inode
        self.fobj.seek(off)
        return await self.fobj.read(size)
