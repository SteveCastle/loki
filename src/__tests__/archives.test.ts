import { isArchivePath } from '../main/archives';

describe('archives', () => {
  describe('isArchivePath', () => {
    it('returns true for .cbz', () => {
      expect(isArchivePath('C:\\comics\\book.cbz')).toBe(true);
      expect(isArchivePath('/home/u/book.cbz')).toBe(true);
    });
    it('returns true for .zip', () => {
      expect(isArchivePath('/tmp/archive.zip')).toBe(true);
    });
    it('is case-insensitive', () => {
      expect(isArchivePath('BOOK.CBZ')).toBe(true);
      expect(isArchivePath('Book.Zip')).toBe(true);
    });
    it('returns false for non-archive paths', () => {
      expect(isArchivePath('/home/u/image.jpg')).toBe(false);
      expect(isArchivePath('/home/u/folder')).toBe(false);
      expect(isArchivePath('/home/u/file.rar')).toBe(false); // not in scope
    });
  });
});
