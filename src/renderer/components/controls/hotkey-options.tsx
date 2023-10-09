import { useState, useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';

import './hotkey-options.css';
export default function HotKeyOptions() {
  const { libraryService } = useContext(GlobalStateContext);
  const hotKeys = useSelector(libraryService, (state) => state.context.hotKeys);

  return (
    <div className="HotKeyOptions">
      {Object.entries(hotKeys).map(([key, value]) => {
        return (
          <div className="HotKeyOption" key={key}>
            <div className="HotKeyOptionKey">{key}</div>
            <div className="HotKeyOptionValue">{value}</div>
          </div>
        );
      })}
    </div>
  );
}
