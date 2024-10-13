import React, { useContext } from 'react';
import { useSelector } from '@xstate/react';
import { useMutation } from '@tanstack/react-query';
import filter from 'renderer/filter';
import { GlobalStateContext, Item } from '../../state';
import './BattleMode.css';

interface Props {
  item: Item;
  offset: number;
}

function calculateElo(
  score1: number,
  score2: number,
  k: number,
  outcome: number
) {
  const expectedScore1 = 1 / (1 + 10 ** ((score2 - score1) / 400));
  const expectedScore2 = 1 / (1 + 10 ** ((score1 - score2) / 400));
  const newScore1 = score1 + k * (outcome - expectedScore1);
  const newScore2 = score2 + k * (1 - outcome - expectedScore2);
  return [newScore1, newScore2];
}

const updateElo = async ({
  winningPath,
  winningElo,
  losingPath,
  losingElo,
}: {
  winningPath: string;
  winningElo: number;
  losingPath: string;
  losingElo: number;
}) => {
  await window.electron.ipcRenderer.invoke('update-elo', [
    winningPath,
    winningElo,
    losingPath,
    losingElo,
  ]);
};

export default function BattleMode({ item, offset }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  console.log(item);
  const { mutate } = useMutation({
    mutationFn: updateElo,
    onSuccess: (data, variables) => {
      libraryService.send('SHUFFLE');
      libraryService.send('UPDATE_MEDIA_ELO', {
        path: variables.winningPath,
        elo: variables.winningElo,
      });
      libraryService.send('UPDATE_MEDIA_ELO', {
        path: variables.losingPath,
        elo: variables.losingElo,
      });
    },
  });

  const otherItemIndex = offset === 0 ? 1 : 0;
  const otherItem = useSelector(
    libraryService,
    (state) =>
      filter(
        state.context.libraryLoadId,
        state.context.textFilter,
        state.context.library,
        state.context.settings.filters,
        state.context.settings.sortBy
      )[state.context.cursor + otherItemIndex],
    (a, b) => a?.path === b?.path && a?.timeStamp === b?.timeStamp
  );

  return (
    <div className={`BattleMode ${offset ? 'left' : 'right'}`}>
      <button
        onClick={() => {
          const score1 = item?.elo || 1500;
          const score2 = otherItem?.elo || 1500;
          const [elo1, elo2] = calculateElo(score1, score2, 48, 1);
          mutate({
            winningPath: item?.path,
            winningElo: elo1,
            losingPath: otherItem?.path,
            losingElo: elo2,
          });
        }}
      >
        <span className="vote-label">Vote</span>
        <span className="elo-score">
          {item?.elo ? item?.elo.toFixed(0) : 'Unranked'}
        </span>
      </button>
    </div>
  );
}
