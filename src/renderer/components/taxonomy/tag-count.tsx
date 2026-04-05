import { useQuery } from '@tanstack/react-query';
import { fetchTagCount } from '../../platform';
import './taxonomy.css';
type Concept = {
  label: string;
  weight: number;
  category: string;
};

type Props = {
  tag: Concept;
};

const fetchTagCountFn = (tag: string) => async (): Promise<number> => {
  const count = await fetchTagCount(tag);
  return count;
};

export default function TagCount({ tag }: Props) {
  const { data: count } = useQuery<number, Error>(
    ['taxonomy', 'tag', tag.label, 'count'],
    fetchTagCountFn(tag.label),
    {
      refetchOnWindowFocus: false,
    }
  );

  return <span>{count}</span>;
}
