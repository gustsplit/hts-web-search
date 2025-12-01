import { useState } from 'react';
import { Calculator } from 'lucide-react';
import type { HTSCandidate } from '../types';

interface DutySimulatorProps {
    selectedHTS: HTSCandidate | null;
    usdKrwRate: number | null;
}

export default function DutySimulator({ selectedHTS, usdKrwRate }: DutySimulatorProps) {
    const [qty, setQty] = useState<number | ''>('');
    const [unitPrice, setUnitPrice] = useState<number | ''>('');
    const [result, setResult] = useState<{
        totalUsd: number;
        dutyUsd: number;
        totalKrw: number;
        dutyKrw: number;
    } | null>(null);

    const handleCalculate = () => {
        if (!selectedHTS || !qty || !unitPrice) return;

        // Parse rate string "4.5%" -> 0.045
        let rate = 0;
        const rateStr = selectedHTS.estimated_tariff_rate || '';
        const match = rateStr.match(/[\d.]+/);
        if (match) {
            rate = parseFloat(match[0]) / 100;
        }

        const totalUsd = Number(qty) * Number(unitPrice);
        const dutyUsd = totalUsd * rate;

        const rateKrw = usdKrwRate || 1400; // Fallback
        const totalKrw = totalUsd * rateKrw;
        const dutyKrw = dutyUsd * rateKrw;

        setResult({ totalUsd, dutyUsd, totalKrw, dutyKrw });
    };

    if (!selectedHTS) return null;

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h3 className="text-lg font-bold mb-2">실제 얼마나 관세가 나올지 궁금하신가요?</h3>
            <p className="text-gray-500 text-sm mb-6">
                선택하신 HTS 코드({selectedHTS.hts_code}) 기준으로 예상 관세를 계산해 보세요.
                {usdKrwRate && <span className="block text-xs mt-1 text-blue-500">현재 적용 환율: 1 USD ≈ {usdKrwRate.toFixed(2)}원</span>}
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div>
                    <label className="block text-sm font-semibold mb-1">수량</label>
                    <input
                        type="number"
                        className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 outline-none"
                        placeholder="예: 100"
                        value={qty}
                        onChange={(e) => setQty(Number(e.target.value))}
                    />
                </div>
                <div>
                    <label className="block text-sm font-semibold mb-1">단가 (USD $)</label>
                    <input
                        type="number"
                        className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 outline-none"
                        placeholder="예: 50"
                        value={unitPrice}
                        onChange={(e) => setUnitPrice(Number(e.target.value))}
                    />
                </div>
                <div className="flex items-end">
                    <button
                        onClick={handleCalculate}
                        className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2.5 px-6 rounded-full inline-flex items-center gap-2 transition shadow-lg shadow-green-600/30 w-full justify-center md:w-auto"
                    >
                        <Calculator className="w-4 h-4" />
                        <span>관세 계산하기</span>
                    </button>
                </div>
            </div>

            {result && (
                <div className="bg-gray-50 rounded-xl p-4 border border-gray-100">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <div className="text-gray-500 mb-1">과세가격 합계 (USD)</div>
                            <div className="font-bold text-lg">${result.totalUsd.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                        </div>
                        <div>
                            <div className="text-gray-500 mb-1">예상 관세 (USD)</div>
                            <div className="font-bold text-lg text-red-600">${result.dutyUsd.toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>
                        </div>
                        <div className="pt-2 border-t border-gray-200">
                            <div className="text-gray-500 mb-1">과세가격 합계 (KRW)</div>
                            <div className="font-medium">{Math.round(result.totalKrw).toLocaleString()}원</div>
                        </div>
                        <div className="pt-2 border-t border-gray-200">
                            <div className="text-gray-500 mb-1">예상 관세 (KRW)</div>
                            <div className="font-medium text-red-600">{Math.round(result.dutyKrw).toLocaleString()}원</div>
                        </div>
                    </div>
                    <p className="text-xs text-gray-400 mt-3 text-center">
                        * 실제 관세는 세관 신고 내용, FTA, 추가 세율 등에 따라 달라질 수 있으며, 이 계산은 참고용 시뮬레이션입니다.
                    </p>
                </div>
            )}
        </div>
    );
}
