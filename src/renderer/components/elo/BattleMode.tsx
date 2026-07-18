import React, { useContext, useMemo } from 'react';
import { useSelector } from '@xstate/react';
import { useMutation, useIsMutating } from '@tanstack/react-query';
import filter from 'renderer/filter';
import { GlobalStateContext, Item } from '../../state';
import { invoke } from '../../platform';
import {
  recordBattleResult,
  recordSkippedPair,
  sortConfidence,
} from '../../battle-pairing';
import './BattleMode.css';

interface Props {
  item: Item;
  offset: number;
}

interface BattleResult {
  winnerPath: string;
  winnerElo: number;
  winnerMatches: number;
  loserPath: string;
  loserElo: number;
  loserMatches: number;
}

// The Elo math runs server-side (Electron main / Go media-server) inside a
// transaction, off possibly-stale client ratings; we just report who won and
// apply the returned ratings. outcome: 1 (default) win, 0.5 draw.
const recordBattle = async ({
  winningPath,
  losingPath,
  outcome,
}: {
  winningPath: string;
  losingPath: string;
  outcome?: number;
}): Promise<BattleResult> => {
  return invoke('record-battle', [winningPath, losingPath, outcome]);
};

// Both panes render a BattleMode instance with their own mutation; the shared
// key lets each pane see the other's in-flight vote and disable its buttons,
// so a fast second click can't double-count a battle.
const BATTLE_MUTATION_KEY = ['record-battle'];

export default function BattleMode({ item, offset }: Props) {
  const { libraryService } = useContext(GlobalStateContext);
  const isBattling =
    useIsMutating({ mutationKey: BATTLE_MUTATION_KEY }) > 0;
  const { mutate } = useMutation({
    mutationKey: BATTLE_MUTATION_KEY,
    mutationFn: recordBattle,
    onSuccess: (data, variables) => {
      if (data) {
        recordBattleResult(
          data.winnerPath,
          data.winnerElo,
          data.loserPath,
          data.loserElo,
          variables.outcome ?? 1
        );
        libraryService.send('UPDATE_MEDIA_ELO', {
          path: data.winnerPath,
          elo: data.winnerElo,
          battles: data.winnerMatches,
        });
        libraryService.send('UPDATE_MEDIA_ELO', {
          path: data.loserPath,
          elo: data.loserElo,
          battles: data.loserMatches,
        });
      }
      libraryService.send('NEXT_BATTLE');
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

  // filter() memoizes, so the ref only changes on reorder (votes, setting
  // changes) — the O(n) confidence scan reruns only then, not per event.
  const view = useSelector(libraryService, (state) =>
    filter(
      state.context.libraryLoadId,
      state.context.textFilter,
      state.context.library,
      state.context.settings.filters,
      state.context.settings.sortBy
    )
  );
  const confidence = useMemo(() => sortConfidence(view), [view]);

  const pairReady = !!item?.path && !!otherItem?.path;

  return (
    <div className={`BattleMode ${offset ? 'left' : 'right'}`}>
      <button
        disabled={!pairReady || isBattling}
        onClick={() => {
          if (!pairReady || isBattling) return;
          mutate({
            winningPath: item.path,
            losingPath: otherItem.path,
          });
        }}
      >
        <span className="vote-label">Vote</span>
        <span className="elo-score">
          {item?.elo ? item?.elo.toFixed(0) : 'Unranked'}
        </span>
      </button>
      {offset === 1 ? (
        // Pair-level controls, rendered once. This instance lives in the
        // physically-right pane with its vote button at the pane's left edge
        // — the seam where both vote buttons meet. Anchoring here (absolute
        // within the pane, shifted half onto the seam) keeps the cluster
        // glued to the vote buttons no matter how other panels resize the
        // detail area.
        <div className="battle-pair-controls">
          <div className="pair-buttons">
            <button
              disabled={!pairReady || isBattling}
              onClick={() => {
                if (!pairReady || isBattling) return;
                mutate({
                  winningPath: item.path,
                  losingPath: otherItem.path,
                  outcome: 0.5,
                });
              }}
            >
              Draw
            </button>
            <button
              disabled={!pairReady || isBattling}
              onClick={() => {
                if (!pairReady || isBattling) return;
                recordSkippedPair(item.path, otherItem.path);
                libraryService.send('NEXT_BATTLE');
              }}
            >
              Skip
            </button>
          </div>
          <span
            className="battle-confidence"
            title={`Sort confidence: mean progress toward ${
              confidence.target
            } battles per item (≈2·log₂ of set size). Avg ${confidence.avgBattles.toFixed(
              1
            )} battles/item · ${confidence.unranked} unranked of ${
              view.length
            }.`}
          >
            <span className="confidence-meter">
              <span
                className="confidence-fill"
                style={{
                  width: `${Math.round(confidence.score * 100)}%`,
                }}
              />
            </span>
            <span className="confidence-value">
              {Math.round(confidence.score * 100)}%
            </span>
          </span>
        </div>
      ) : null}
    </div>
  );
}
