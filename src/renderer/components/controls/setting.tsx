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
                    // If current value is less than 5 increment is 1, otherwise increment is option value to give more fine grained control at small numbers.
                    const increment =
                      currentValue <= 5 ? 1 : option.increment ?? 1;

                    const newValue =
                      (currentValue as number) - (increment ?? 1);
                    const minValue = (option as any).min ?? 0;

                    if (newValue < minValue) return;

                    libraryService.send('CHANGE_SETTING', {
                      data: {
                        [settingKey]: newValue,
                      },
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
                    const increment =
                      currentValue < 5 ? 1 : option.increment ?? 1;

                    const newValue =
                      (currentValue as number) + (increment ?? 1);
                    const maxValue = (option as any).max;

                    // Check if max value is defined and new value would exceed it
                    if (maxValue !== undefined && newValue > maxValue) return;

                    libraryService.send('CHANGE_SETTING', {
                      data: {
                        [settingKey]: newValue,
                      },
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
