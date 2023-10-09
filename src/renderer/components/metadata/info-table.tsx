import React from 'react';
import moment from 'moment';
import { FileMetadata, StableDiffusionMetaData } from '../../../main/metadata';
import copyIcon from '../../../../assets/copy.svg';

import './info-table.css';

type Props = {
  data: FileMetadata | StableDiffusionMetaData;
};

const InfoTable: React.FC<Props> = ({ data }) => {
  return (
    <div className="InfoTable">
      <table>
        <tbody>
          {Object.entries(data).map(([key, value]) => {
            let formattedValue;
            if (typeof value === 'boolean') {
              formattedValue = value ? '✔️' : '❌';
            } else if (
              ['modified'].includes(key) &&
              moment(value, moment.ISO_8601, true).isValid()
            ) {
              formattedValue = moment(value).format('MMMM Do YYYY, h:mm:ss a');
            } else {
              formattedValue = value;
            }

            return (
              <tr key={key}>
                <td style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                  {key}
                </td>
                <td>
                  <div className="value">
                    {formattedValue}
                    <button
                      className="copy-button"
                      onClick={() => {
                        const copyContent = async (text: string) => {
                          try {
                            await navigator.clipboard.writeText(text);
                            console.log('Content copied to clipboard');
                          } catch (err) {
                            console.error('Failed to copy: ', err);
                          }
                        };
                        copyContent(value);
                      }}
                    >
                      <img src={copyIcon} />
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default InfoTable;
