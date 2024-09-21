export enum MediaTypes {
  Static = 'static',
  Motion = 'motion',
  All = 'all',
}

export enum FileTypes {
  Image = 'image',
  Video = 'video',
  Audio = 'audio',
  Document = 'document',
  Other = 'other',
}

export enum Extensions {
  Image = 'jpg|jpeg|png|gif|bmp|svg|jfif|pjpeg|pjp|webp',
  Video = 'mov|mp4|webm|ogg|mkv',
  Audio = 'mp3|wav',
  Document = 'pdf|doc|docx|xls|xlsx|ppt|pptx|txt|csv',
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
    if (Extensions.Image.includes(extension)) {
      return FileTypes.Image;
    }
    if (Extensions.Video.includes(extension)) {
      return FileTypes.Video;
    }
    if (Extensions.Audio.includes(extension)) {
      return FileTypes.Audio;
    }
    if (Extensions.Document.includes(extension)) {
      return FileTypes.Document;
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
  return MediaTypes.All;
};
