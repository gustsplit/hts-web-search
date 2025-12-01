export interface HTSRequest {
    product_name: string;
    exporter_hs?: string;
    description: string;
    country_of_origin: string;
}

export interface HTSCandidate {
    hts_code: string;
    title: string;
    estimated_tariff_rate?: string;
    reason?: string;
    confidence?: number;
}

export interface DutyCalculationResult {
    qty: number;
    unit_price: number;
    total_value_usd: number;
    duty_usd: number;
    total_value_krw?: number;
    duty_krw?: number;
    usd_krw_rate?: number;
}
