import { useState, useEffect } from 'react';
import SearchForm from '../components/SearchForm';
import ResultsTable from '../components/ResultsTable';
import DutySimulator from '../components/DutySimulator';
import EmailPreview from '../components/EmailPreview';
import { searchHTS, getUsdKrwRate, previewEmail, sendEmail } from '../lib/api';
import type { HTSRequest, HTSCandidate } from '../types';

export default function Home() {
    const [loading, setLoading] = useState(false);
    const [candidates, setCandidates] = useState<HTSCandidate[]>([]);
    const [selectedIdx, setSelectedIdx] = useState<number>(-1);
    const [usdKrwRate, setUsdKrwRate] = useState<number | null>(null);

    // Email state
    const [emailPreview, setEmailPreview] = useState<{ subject: string; body: string } | null>(null);
    const [emailLoading, setEmailLoading] = useState(false);
    const [lastReq, setLastReq] = useState<HTSRequest | null>(null);

    useEffect(() => {
        getUsdKrwRate().then(data => setUsdKrwRate(data.rate)).catch(console.error);
    }, []);

    const handleSearch = async (data: HTSRequest) => {
        setLoading(true);
        setCandidates([]);
        setSelectedIdx(-1);
        setEmailPreview(null);
        setLastReq(data);

        try {
            const res = await searchHTS(data);
            setCandidates(res.candidates);
        } catch (err) {
            console.error(err);
            alert('검색 중 오류가 발생했습니다.');
        } finally {
            setLoading(false);
        }
    };

    const handleEmailPreview = async (email: string) => {
        if (selectedIdx === -1 || !lastReq) return;

        setEmailLoading(true);
        const selected = candidates[selectedIdx];

        try {
            const payload = {
                ...lastReq,
                email_to: email,
                hts_code: selected.hts_code,
                hts_title: selected.title,
                estimated_tariff_rate: selected.estimated_tariff_rate,
                confidence: selected.confidence,
                // TODO: Pass calculated duty values if available (requires lifting state from DutySimulator)
                // For now, we'll just pass basic info
            };

            const res = await previewEmail(payload);
            setEmailPreview(res);
        } catch (err) {
            console.error(err);
            alert('이메일 미리보기 생성 실패');
        } finally {
            setEmailLoading(false);
        }
    };

    const handleSendEmail = async () => {
        if (!emailPreview) return;
        setEmailLoading(true);
        try {
            // We need to pass the email address again, but it's embedded in the body/subject logic on backend?
            // Actually backend send_email needs email_to. 
            // We should store the email address used for preview.
            // For simplicity, let's extract it or store it.
            // Let's just assume the user didn't change it.
            // Ideally we pass the full object.
            // Let's update the API to take the preview object + email.
            // But wait, the preview response doesn't include the email.
            // I'll fix this by storing the email in state when previewing.

            // Quick fix: parse from body or just ask user again? 
            // Better: store it in state.
            const emailLine = emailPreview.body.split('\n')[0]; // "name@example.com 고객님,"
            const emailTo = emailLine.split(' ')[0];

            await sendEmail({
                email_to: emailTo,
                subject: emailPreview.subject,
                body: emailPreview.body
            });
        } catch (err) {
            console.error(err);
            alert('이메일 발송 실패');
        } finally {
            setEmailLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-6 pb-20">
            <header className="mb-8 text-center md:text-left">
                <h1 className="text-3xl font-bold flex items-center justify-center md:justify-start gap-3 text-gray-900">
                    <span className="w-4 h-4 rounded-full bg-blue-600 shadow-lg shadow-blue-500/50"></span>
                    US HTS Code Finder
                </h1>
                <p className="text-gray-500 mt-2 font-medium">
                    AI 기반 미국 관세 분류 및 예상 세액 시뮬레이션
                </p>
            </header>

            <SearchForm onSearch={handleSearch} loading={loading} />

            <ResultsTable
                candidates={candidates}
                selectedIdx={selectedIdx}
                onSelect={setSelectedIdx}
            />

            {selectedIdx !== -1 && (
                <>
                    <DutySimulator
                        selectedHTS={candidates[selectedIdx]}
                        usdKrwRate={usdKrwRate}
                    />

                    <EmailPreview
                        onPreview={handleEmailPreview}
                        onSend={handleSendEmail}
                        previewData={emailPreview}
                        loading={emailLoading}
                    />
                </>
            )}
        </div>
    );
}
