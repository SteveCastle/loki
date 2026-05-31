export enum MediaTypes {
  Static = 'static',
  Motion = 'motion',
  Audio = 'audio',
  All = 'all',
}

export enum FileTypes {
  Image = 'image',
  Video = 'video',
  Audio = 'audio',
  Document = 'document',
  Archive = 'archive',
  Other = 'other',
}

export enum Extensions {
  Image = 'jpg|jpeg|png|gif|bmp|svg|jfif|pjpeg|pjp|webp|avif',
  Video = 'mov|mp4|webm|ogg|mkv|m4v',
  Audio = 'mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape',
  Document = 'pdf|doc|docx|xls|xlsx|ppt|pptx|txt|csv',
  Archive = 'cbz|zip|cbr',
}

export const getFileType = (
  fileName: string,
  gifIsVideo?: boolean
): FileTypes => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  if (extension) {
    if (gifIsVideo && extension === 'gif') {
      return FileTypes.Video;
    }
    // Exact extension match (split on '|') rather than substring `includes`,
    // so e.g. a `.avi` file is not matched by the `avif` entry.
    if (Extensions.Image.split('|').includes(extension)) {
      return FileTypes.Image;
    }
    if (Extensions.Video.split('|').includes(extension)) {
      return FileTypes.Video;
    }
    if (Extensions.Audio.split('|').includes(extension)) {
      return FileTypes.Audio;
    }
    if (Extensions.Document.split('|').includes(extension)) {
      return FileTypes.Document;
    }
    if (Extensions.Archive.split('|').includes(extension)) {
      return FileTypes.Archive;
    }
  }
  return FileTypes.Other;
};

export const getMediaType = (fileName: string): MediaTypes => {
  const fileType = getFileType(fileName, true);
  if (fileType === FileTypes.Image) {
    return MediaTypes.Static;
  }
  if (fileType === FileTypes.Video) {
    return MediaTypes.Motion;
  }
  if (fileType === FileTypes.Audio) {
    return MediaTypes.Audio;
  }
  return MediaTypes.All;
};
