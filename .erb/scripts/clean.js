import * as rimraf from 'rimraf';
import webpackPaths from '../configs/webpack.paths';

const foldersToRemove = [
  webpackPaths.distPath,
  webpackPaths.buildPath,
  webpackPaths.dllPath,
];

foldersToRemove.forEach((folder) => {
  try {
    rimraf.sync(folder);
  } catch (error) {
    // On Windows, files can be locked (EBUSY/EPERM).
    // Log warning but continue - files will be overwritten during build anyway.
    if (error.code === 'EBUSY' || error.code === 'EPERM') {
      console.warn(
        `Warning: Could not delete ${folder} - file may be locked. ` +
          `This is usually fine as files will be overwritten during build. ` +
          `If build fails, close any applications using files in this directory.`
      );
    } else {
      // Re-throw other errors
      throw error;
    }
  }
});
