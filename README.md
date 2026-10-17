# MRI Detection

## Accessing a dataset in Google Drive from VS Code

This project runs locally, so Colab's `google.colab.drive.mount()` is not available. Use [rclone](https://rclone.org/drive/) to mount Google Drive as a local folder that the notebook can read.

1. Install `rclone` and FUSE for your Linux distribution.
2. In a terminal, run `rclone config` and create a remote named `gdrive` with storage type **Google Drive**. Follow the browser sign-in/authorization prompts. rclone keeps its OAuth credentials in your home-directory config, not in this project.
3. Check the Drive folder names with:

   ```sh
   rclone lsd gdrive:
   ```

4. Create the default mount point and mount the folder containing your dataset (replace `MRI Dataset` with the actual folder path in your Drive):

   ```sh
   mkdir -p data
   rclone mount 'gdrive:MRI Dataset' "$PWD/data" --vfs-cache-mode full
   ```

   Keep this command running while using the notebook. To disconnect, press **Ctrl+C** in that terminal. If the dataset is in a nested folder, include its path after `gdrive:`, for example `'gdrive:Datasets/MRI'`.
5. Open `project_code.ipynb` in VS Code and run the **Google Drive dataset** cell. It sets `DATA_DIR` to the mounted folder and lists its top-level items. Use `DATA_DIR` as the root for the project's image/data loading code.

If you mount somewhere other than the project's `data/` folder, set `MRI_DATA_DIR` to that mount path before starting the notebook kernel. The local `data/` folder is excluded from Git so the dataset itself is not committed.
