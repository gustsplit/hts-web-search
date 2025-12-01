import type { HTSCandidate } from '../types';

interface ResultsTableProps {
    candidates: HTSCandidate[];
    selectedIdx: number;
    onSelect: (idx: number) => void;
}

export default function ResultsTable({ candidates, selectedIdx, onSelect }: ResultsTableProps) {
    if (candidates.length === 0) return null;

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                HTS 코드 추천 결과
                <span className="text-xs font-normal bg-blue-100 text-blue-600 px-2 py-1 rounded-full">LLM 기반 후보 리스트</span>
            </h3>

            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 text-gray-600 font-medium">
                        <tr>
                            <th className="p-3 rounded-tl-lg">선택</th>
                            <th className="p-3">신뢰도</th>
                            <th className="p-3">HTS Code</th>
                            <th className="p-3">품목명</th>
                            <th className="p-3">관세율</th>
                            <th className="p-3 rounded-tr-lg">분류 설명</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                        {candidates.map((c, idx) => (
                            <tr
                                key={idx}
                                className={`hover:bg-blue-50 transition cursor-pointer ${selectedIdx === idx ? 'bg-blue-50' : ''}`}
                                onClick={() => onSelect(idx)}
                            >
                                <td className="p-3">
                                    <input
                                        type="radio"
                                        name="hts_choice"
                                        checked={selectedIdx === idx}
                                        onChange={() => onSelect(idx)}
                                        className="accent-blue-500"
                                    />
                                </td>
                                <td className="p-3 font-medium text-blue-600">{c.confidence ? `${c.confidence}%` : '-'}</td>
                                <td className="p-3 font-mono font-bold text-gray-900">{c.hts_code}</td>
                                <td className="p-3 text-gray-800">{c.title}</td>
                                <td className="p-3 text-gray-600">{c.estimated_tariff_rate}</td>
                                <td className="p-3 text-gray-500 text-xs leading-relaxed max-w-xs">{c.reason}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
