import os
import shutil
import zipfile as zf
import errno



def unzip_and_remove(zip_filename):
    if not os.path.exists(zip_filename):
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            zip_filename,
        )
    files = zf.ZipFile(zip_filename, 'r')
    files.extractall('mlsmm2156')
    files.close()
    os.remove(zip_filename)
    shutil.rmtree('mlsmm2156/__MACOSX')  # remove macosx data for Mac users


if __name__ == "__main__":
    unzip_and_remove("mlsmm2156/data.zip")
