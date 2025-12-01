import type { HTSRequest, HTSCandidate } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function searchHTS(data: HTSRequest): Promise<{ candidates: HTSCandidate[] }> {
    const res = await fetch(`${API_BASE_URL}/api/search_hts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to search HTS codes');
    return res.json();
}

export async function getUsdKrwRate(): Promise<{ rate: number }> {
    const res = await fetch(`${API_BASE_URL}/api/exchange-rate/usd-krw`);
    if (!res.ok) throw new Error('Failed to fetch exchange rate');
    return res.json();
}

export async function previewEmail(data: any): Promise<{ subject: string; body: string }> {
    const res = await fetch(`${API_BASE_URL}/api/preview_email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to preview email');
    return res.json();
}

export async function sendEmail(data: { email_to: string; subject: string; body: string }): Promise<void> {
    const res = await fetch(`${API_BASE_URL}/api/send_email`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to send email');
}
