import {
  getFileType,
  getMediaType,
  FileTypes,
  MediaTypes,
  Extensions,
} from '../file-types';

describe('file-types', () => {
  describe('getFileType', () => {
    describe('image files', () => {
      const imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'jfif', 'pjpeg', 'pjp', 'webp'];

      it.each(imageExtensions)('should identify .%s as Image', (ext) => {
        expect(getFileType(`photo.${ext}`)).toBe(FileTypes.Image);
        expect(getFileType(`PHOTO.${ext.toUpperCase()}`)).toBe(FileTypes.Image);
      });

      it('should identify gif as Image by default', () => {
        expect(getFileType('animation.gif')).toBe(FileTypes.Image);
      });

      it('should identify gif as Video when gifIsVideo is true', () => {
        expect(getFileType('animation.gif', true)).toBe(FileTypes.Video);
      });
    });

    describe('video files', () => {
      const videoExtensions = ['mov', 'mp4', 'webm', 'ogg', 'mkv', 'm4v'];

      it.each(videoExtensions)('should identify .%s as Video', (ext) => {
        expect(getFileType(`movie.${ext}`)).toBe(FileTypes.Video);
        expect(getFileType(`MOVIE.${ext.toUpperCase()}`)).toBe(FileTypes.Video);
      });
    });

    describe('audio files', () => {
      const audioExtensions = ['mp3', 'wav', 'flac', 'aac', 'm4a', 'opus', 'wma', 'aiff', 'ape'];

      it.each(audioExtensions)('should identify .%s as Audio', (ext) => {
        expect(getFileType(`song.${ext}`)).toBe(FileTypes.Audio);
        expect(getFileType(`SONG.${ext.toUpperCase()}`)).toBe(FileTypes.Audio);
      });
    });

    describe('document files', () => {
      const docExtensions = ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'csv'];

      it.each(docExtensions)('should identify .%s as Document', (ext) => {
        expect(getFileType(`document.${ext}`)).toBe(FileTypes.Document);
      });
    });

    describe('unknown files', () => {
      it('should return Other for unknown extensions', () => {
        expect(getFileType('file.xyz')).toBe(FileTypes.Other);
        expect(getFileType('file.unknown')).toBe(FileTypes.Other);
        expect(getFileType('file.exe')).toBe(FileTypes.Other);
      });

      it('should return Other for files without extension', () => {
        expect(getFileType('noextension')).toBe(FileTypes.Other);
      });

      it('should handle files with multiple dots', () => {
        expect(getFileType('photo.backup.jpg')).toBe(FileTypes.Image);
        expect(getFileType('video.2024.01.01.mp4')).toBe(FileTypes.Video);
      });
    });

    describe('edge cases', () => {
      it('should handle empty string', () => {
        expect(getFileType('')).toBe(FileTypes.Other);
      });

      it('should handle path with directories', () => {
        expect(getFileType('/home/user/photos/image.jpg')).toBe(FileTypes.Image);
        expect(getFileType('C:\\Users\\Photos\\image.png')).toBe(FileTypes.Image);
      });

      it('should handle hidden files', () => {
        expect(getFileType('.hidden.jpg')).toBe(FileTypes.Image);
      });
    });
  });

  describe('getMediaType', () => {
    it('should return Static for image files', () => {
      expect(getMediaType('photo.jpg')).toBe(MediaTypes.Static);
      expect(getMediaType('photo.png')).toBe(MediaTypes.Static);
      expect(getMediaType('photo.webp')).toBe(MediaTypes.Static);
    });

    it('should return Motion for video files', () => {
      expect(getMediaType('video.mp4')).toBe(MediaTypes.Motion);
      expect(getMediaType('video.mov')).toBe(MediaTypes.Motion);
      expect(getMediaType('video.webm')).toBe(MediaTypes.Motion);
    });

    it('should return Motion for gif files (gifIsVideo=true internally)', () => {
      expect(getMediaType('animation.gif')).toBe(MediaTypes.Motion);
    });

    it('should return Audio for audio files', () => {
      expect(getMediaType('song.mp3')).toBe(MediaTypes.Audio);
      expect(getMediaType('song.wav')).toBe(MediaTypes.Audio);
      expect(getMediaType('song.flac')).toBe(MediaTypes.Audio);
    });

    it('should return All for unknown/other files', () => {
      expect(getMediaType('file.exe')).toBe(MediaTypes.All);
      expect(getMediaType('file.unknown')).toBe(MediaTypes.All);
      expect(getMediaType('document.pdf')).toBe(MediaTypes.All);
    });
  });

  describe('Extensions enum', () => {
    it('should contain common image extensions', () => {
      expect(Extensions.Image).toContain('jpg');
      expect(Extensions.Image).toContain('png');
      expect(Extensions.Image).toContain('webp');
    });

    it('should contain common video extensions', () => {
      expect(Extensions.Video).toContain('mp4');
      expect(Extensions.Video).toContain('mov');
      expect(Extensions.Video).toContain('webm');
    });

    it('should contain common audio extensions', () => {
      expect(Extensions.Audio).toContain('mp3');
      expect(Extensions.Audio).toContain('wav');
      expect(Extensions.Audio).toContain('flac');
    });
  });

  describe('MediaTypes enum', () => {
    it('should have correct values', () => {
      expect(MediaTypes.Static).toBe('static');
      expect(MediaTypes.Motion).toBe('motion');
      expect(MediaTypes.Audio).toBe('audio');
      expect(MediaTypes.All).toBe('all');
    });
  });

  describe('FileTypes enum', () => {
    it('should have correct values', () => {
      expect(FileTypes.Image).toBe('image');
      expect(FileTypes.Video).toBe('video');
      expect(FileTypes.Audio).toBe('audio');
      expect(FileTypes.Document).toBe('document');
      expect(FileTypes.Other).toBe('other');
    });
  });
});
