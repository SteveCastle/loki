import React, { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import './cache-setting.css';

function CacheSetting() {
  const { libraryService } = useContext(GlobalStateContext);
  const { listImageCache, detailImageCache } = useSelector(
    libraryService,
    (state) => state.context.settings
  );

  return (
    <div className={'CacheSetting'}>
      <button
        onClick={() => {
          libraryService.send('CHANGE_SETTING', {
            data: {
              listImageCache: listImageCache ? false : 'thumbnail_path_600',
            },
          });
        }}
      >
        {listImageCache ? 'Disable List Cache' : 'Enable List Cache'}
      </button>
      <button
        onClick={() => {
          libraryService.send('CHANGE_SETTING', {
            data: {
              detailImageCache: detailImageCache
                ? false
                : 'thumbnail_path_1200',
            },
          });
        }}
      >
        {detailImageCache ? 'Disable Detail Cache' : 'Enable Detail Cache'}
      </button>
    </div>
  );
}

export default CacheSetting;
