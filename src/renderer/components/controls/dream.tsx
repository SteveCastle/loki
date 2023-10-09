import { useState, useEffect } from 'react';
import './dream.css';

export default function Dream() {
  // an array of dream like emojii characters
  const [char, setChar] = useState('🌠');

  const emojii = [
    '🌈',
    '🌠',
    '🪐',
    '🌇',
    '🦄',
    '🌌',
    '🌄',
    '🧜‍♀️',
    '🌆',
    '🌉',
    '🌊',
    '🌋',
    '🌅',
    '🌸',
    '🌌',
    '🌈',
    '🌙',
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setChar((char) => {
        const index = emojii.indexOf(char);
        if (index === emojii.length - 1) {
          return emojii[0];
        } else {
          return emojii[index + 1];
        }
      });
    }, 750);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container">
      <span className="cloud-emoji">{char}</span>
    </div>
  );
}
