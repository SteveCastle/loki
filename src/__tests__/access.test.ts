import { deriveCanWrite } from '../renderer/access';

describe('deriveCanWrite', () => {
  it('signed-in users can always write', () => {
    expect(deriveCanWrite({ loggedIn: true, publicAccess: true })).toBe(true);
    expect(deriveCanWrite({ loggedIn: true, publicAccess: false })).toBe(true);
  });

  it('anonymous visitors on a public-access server are view-only', () => {
    expect(deriveCanWrite({ loggedIn: false, publicAccess: true })).toBe(false);
  });

  it('anonymous + flag off stays permissive (server 302s /app/ to /login, so this state never renders; permissive preserves the 401→/login flow)', () => {
    expect(deriveCanWrite({ loggedIn: false, publicAccess: false })).toBe(true);
  });
});
