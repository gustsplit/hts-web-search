import { useState } from 'react';
import { Mail, Send, CheckCircle } from 'lucide-react';

interface EmailPreviewProps {
    onPreview: (email: string) => Promise<void>;
    onSend: () => Promise<void>;
    previewData: { subject: string; body: string } | null;
    loading: boolean;
}

export default function EmailPreview({ onPreview, onSend, previewData, loading }: EmailPreviewProps) {
    const [email, setEmail] = useState('');
    const [sent, setSent] = useState(false);

    const handlePreview = async () => {
        if (!email) return;
        await onPreview(email);
    };

    const handleSend = async () => {
        await onSend();
        setSent(true);
    };

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <h3 className="text-lg font-bold mb-2 flex items-center gap-2">
                이메일로 결과 받아보기
                <span className="text-xs font-normal bg-blue-100 text-blue-600 px-2 py-1 rounded-full">Step 3</span>
            </h3>
            <p className="text-gray-500 text-sm mb-6">
                입력하신 상품 정보, 선택하신 HTS 코드, 그리고 계산된 관세 금액을 정리해서 메일로 보내드려요.
            </p>

            <div className="flex flex-col md:flex-row gap-4 items-end mb-6">
                <div className="w-full md:w-2/3">
                    <label className="block text-sm font-semibold mb-1">이메일 주소</label>
                    <input
                        type="email"
                        className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 outline-none"
                        placeholder="name@example.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />
                </div>
                <button
                    onClick={handlePreview}
                    disabled={loading || !email}
                    className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2.5 px-6 rounded-full inline-flex items-center gap-2 transition shadow-lg shadow-blue-500/30 disabled:opacity-70 whitespace-nowrap"
                >
                    {loading ? <Loader2 className="animate-spin w-4 h-4" /> : <Mail className="w-4 h-4" />}
                    <span>이메일 미리보기</span>
                </button>
            </div>

            {previewData && (
                <div className="bg-gray-50 rounded-xl p-6 border border-gray-200 animate-in fade-in zoom-in-95 duration-300">
                    <div className="mb-4 pb-4 border-b border-gray-200">
                        <div className="text-sm text-gray-500 mb-1">받는 사람: {email}</div>
                        <div className="font-bold text-gray-900">{previewData.subject}</div>
                    </div>
                    <div className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed font-sans">
                        {previewData.body}
                    </div>

                    <div className="mt-6 flex justify-end">
                        {!sent ? (
                            <button
                                onClick={handleSend}
                                disabled={loading}
                                className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2.5 px-8 rounded-full inline-flex items-center gap-2 transition shadow-lg shadow-green-600/30"
                            >
                                <Send className="w-4 h-4" />
                                <span>이 내용으로 발송하기</span>
                            </button>
                        ) : (
                            <div className="flex items-center gap-2 text-green-600 font-bold bg-green-50 px-4 py-2 rounded-full">
                                <CheckCircle className="w-5 h-5" />
                                <span>발송되었습니다!</span>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

import { Loader2 } from 'lucide-react';
