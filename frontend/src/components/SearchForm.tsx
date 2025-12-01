import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';
import type { HTSRequest } from '../types';

interface SearchFormProps {
    onSearch: (data: HTSRequest) => Promise<void>;
    loading: boolean;
}

export default function SearchForm({ onSearch, loading }: SearchFormProps) {
    const [formData, setFormData] = useState<HTSRequest>({
        product_name: '',
        exporter_hs: '',
        description: '',
        country_of_origin: '',
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSearch(formData);
    };

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label className="block text-sm font-semibold mb-1">제품명 (Product Name)</label>
                    <input
                        type="text"
                        className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition"
                        placeholder="예: 옥수수"
                        required
                        value={formData.product_name}
                        onChange={(e) => setFormData({ ...formData, product_name: e.target.value })}
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-semibold mb-1">수출자 HS 코드 (한국, 선택)</label>
                        <input
                            type="text"
                            className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition"
                            placeholder="예: 1005.90"
                            value={formData.exporter_hs}
                            onChange={(e) => setFormData({ ...formData, exporter_hs: e.target.value })}
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-semibold mb-1">원산지 (Country of Origin)</label>
                        <input
                            type="text"
                            className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition"
                            placeholder="예: Korea"
                            required
                            value={formData.country_of_origin}
                            onChange={(e) => setFormData({ ...formData, country_of_origin: e.target.value })}
                        />
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-semibold mb-1">제품 설명 (Product Description)</label>
                    <textarea
                        className="w-full p-2.5 rounded-lg border border-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition min-h-[80px]"
                        placeholder="예: 조리하지 않은 날것의 식용 옥수수"
                        required
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    />
                </div>

                <button
                    type="submit"
                    disabled={loading}
                    className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2.5 px-6 rounded-full inline-flex items-center gap-2 transition shadow-lg shadow-blue-500/30 disabled:opacity-70"
                >
                    {loading ? <Loader2 className="animate-spin w-4 h-4" /> : <Search className="w-4 h-4" />}
                    <span>HTS 코드 검색하기</span>
                </button>
            </form>
        </div>
    );
}
