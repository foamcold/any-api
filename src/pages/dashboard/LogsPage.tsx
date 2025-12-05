import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { format } from 'date-fns';
import { toZonedTime } from 'date-fns-tz';
import { API_BASE_URL } from '@/utils/api';
import { cn } from '@/lib/utils';
import MaskedKey from '@/components/MaskedKey';
import { Pagination } from '@/components/ui/pagination';
import { useToast } from '@/hooks/useToast';

interface Log {
    id: number;
    model: string;
    status: string;
    status_code: number;
    latency: number;
    ttft: number;
    is_stream: boolean;
    input_tokens: number;
    output_tokens: number;
    created_at: string;
    exclusive_key_key: string;
    official_key_key: string;
}

interface PaginatedResponse<T> {
    total: number;
    items: T[];
    page: number;
    size: number;
}

export default function LogsPage() {
    const [logData, setLogData] = useState<PaginatedResponse<Log>>({ items: [], total: 0, page: 1, size: 20 });
    useState(false);
    const { toast } = useToast();

    const fetchLogs = useCallback(async (page = 1, size = 20) => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get<PaginatedResponse<Log>>(`${API_BASE_URL}/logs/`, {
                headers: { Authorization: `Bearer ${token}` },
                params: { page, size }
            });
            setLogData(response.data);
        } catch (error) {
            console.error('Failed to fetch logs', error);
            toast({ variant: 'error', title: '加载日志失败' });
        } finally {
        }
    }, [toast]);

    useEffect(() => {
        fetchLogs(1, 20);
    }, [fetchLogs]);

    return (
        <div className="space-y-6">

            <div className="bg-card border rounded-lg overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-muted text-muted-foreground">
                            <tr>
                                <th className="p-4 font-medium whitespace-nowrap">时间</th>
                                <th className="p-4 font-medium whitespace-nowrap">模型</th>
                                <th className="p-4 font-medium whitespace-nowrap">密钥</th>
                                <th className="p-4 font-medium whitespace-nowrap">状态</th>
                                <th className="p-4 font-medium whitespace-nowrap">延迟</th>
                                <th className="p-4 font-medium whitespace-nowrap">首Token时间</th>
                                <th className="p-4 font-medium whitespace-nowrap">令牌 (输入/输出)</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y">
                            {logData.items.length === 0 ? (
                                <tr>
                                    <td colSpan={7} className="p-8 text-center text-muted-foreground">
                                        未找到日志。
                                    </td>
                                </tr>
                            ) : (
                                logData.items.map((log) => (
                                    <tr key={log.id} className="hover:bg-accent/50">
                                        <td className="p-4 whitespace-nowrap">
                                            {format(toZonedTime(new Date(log.created_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss')}
                                        </td>
                                        <td className="p-4">{log.model}</td>
                                        <td className="p-4 font-mono text-xs" title={log.exclusive_key_key || log.official_key_key}>
                                            <MaskedKey apiKey={log.exclusive_key_key || log.official_key_key || ''} />
                                        </td>
                                        <td className="p-4">
                                            <span className={cn(
                                                "px-2 py-1 rounded text-xs",
                                                log.status === 'ok' ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" :
                                                    "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                                            )}>
                                                {log.status_code || log.status}
                                            </span>
                                        </td>
                                        <td className="p-4">{log.latency.toFixed(2)}s</td>
                                        <td className="p-4">{log.ttft > 0 ? log.ttft.toFixed(2) + 's' : '-'}</td>
                                        <td className="p-4">
                                            {log.input_tokens} / {log.output_tokens}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <Pagination
                currentPage={logData.page}
                totalPages={Math.ceil(logData.total / logData.size)}
                pageSize={logData.size}
                totalItems={logData.total}
                onPageChange={(page) => fetchLogs(page, logData.size)}
                onPageSizeChange={(size) => fetchLogs(1, size)}
            />
        </div>
    );
}
