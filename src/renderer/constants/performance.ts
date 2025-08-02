// Performance-related constants
export const PERFORMANCE_CONSTANTS = {
  // Virtualization
  LIST_OVERSCAN: 2,
  
  // Animation and scrolling
  SCROLL_SPEED_THRESHOLD: 200,
  SCROLL_SPEED_RANGE: {
    MIN: -200,
    MAX: 200,
  },
  
  // Debounce timers (milliseconds)
  RESIZE_DEBOUNCE: 100,
  SCROLL_DEBOUNCE: 16, // ~60fps
  
  // Cache settings
  QUERY_STALE_TIME: 5 * 60 * 1000, // 5 minutes
  QUERY_CACHE_TIME: 10 * 60 * 1000, // 10 minutes
  
  // File type checking
  SUPPORTED_IMAGE_EXTENSIONS: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg', 'jfif', 'pjpeg', 'pjp', 'webp'],
  SUPPORTED_VIDEO_EXTENSIONS: ['mp4', 'mov', 'mkv', 'webm', 'flv', 'm4v'],
  SUPPORTED_AUDIO_EXTENSIONS: ['mp3', 'wav', 'm4a', 'aac', 'ogg'],
} as const;

// Memoization helpers
export const createMemoKey = (...args: (string | number | boolean | null | undefined)[]) => {
  return args.filter(arg => arg !== null && arg !== undefined).join('-');
};