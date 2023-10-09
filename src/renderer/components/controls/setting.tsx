import React, { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext, Item } from '../../state';
import { SETTINGS, SettingKey } from 'settings';
import './setting.css';

type Props = {
  settingKey: SettingKey;
  reload?: boolean;
  resetCursor?: boolean;
  currentItem: Item;
};

function Setting({
  settingKey,
  reload = true,
  resetCursor = false,
  currentItem,
}: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const currentValue = useSelector(libraryService, (state) => {
    return state.context.settings[settingKey];
  });
  return (
    <div className="Setting">
      <div className="menuLabel">{SETTINGS[settingKey].title}</div>
      <div className="menuSection">
        {Object.values(SETTINGS[settingKey].options).map((option) => (
          <button
            key={option.label}
            className={`menuButton ${
              option.value === currentValue ? 'selected' : ''
            }`}
            onClick={() => {
              libraryService.send(
                reload ? 'CHANGE_SETTING_AND_RELOAD' : 'CHANGE_SETTING',
                {
                  data: { [settingKey]: option.value },
                }
              );
              if (resetCursor)
                libraryService.send('RESET_CURSOR', { currentItem });
            }}
          >
            {typeof option.value === 'number' &&
            typeof currentValue === 'number' ? (
              <div className="numericControls">
                <div
                  className={'incrementButton'}
                  onClick={(e) => {
                    e.stopPropagation();
                    libraryService.send('CHANGE_SETTING', {
                      data: { [settingKey]: (currentValue as number) - 10 },
                    });
                  }}
                >
                  -
                </div>
                <span>{currentValue}</span>
                <div
                  className={'incrementButton'}
                  onClick={(e) => {
                    e.stopPropagation();
                    libraryService.send('CHANGE_SETTING', {
                      data: { [settingKey]: (currentValue as number) + 10 },
                    });
                  }}
                >
                  +
                </div>
              </div>
            ) : (
              option.label
            )}
          </button>
        ))}
      </div>
    </div>
  );
}

export default Setting;
