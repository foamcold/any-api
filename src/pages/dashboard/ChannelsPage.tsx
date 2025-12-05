import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '@/utils/api';
import { Plus, Trash2, Copy, RefreshCw, Activity, AlertCircle, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from '@/components/ui/dialog';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/hooks/useToast';
import { cn } from '@/lib/utils';
import { confirm } from '@/components/ui/ConfirmDialog';
import { Pagination } from '@/components/ui/pagination';
import { Card, CardContent } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import MaskedKey from '@/components/MaskedKey';

interface Channel {
    id: number;
    name: string;
    type: string;
    api_url: string;
    user_id: number;
    created_at: string;
    total_keys: number;
    active_keys: number;
    usage_count: number;
    error_count: number;
    total_tokens: number;
}

interface OfficialKey {
    id: number;
    key: string;
    is_active: boolean;
    usage_count: number;
    error_count: number;
    input_tokens: number;
    output_tokens: number;
    last_status: string | number;
    last_status_code: number | null;
    channel_id: number | null;
    created_at: string;
}

interface PaginatedResponse<T> {
    total: number;
    items: T[];
    page: number;
    size: number;
}

const DEFAULT_CHANNEL_TYPES = [
    { value: 'gemini', label: 'Gemini', defaultUrl: 'https://generativelanguage.googleapis.com' },
    { value: 'openai', label: 'OpenAI', defaultUrl: 'https://api.openai.com' },
];

export default function ChannelsPage() {
    // Data State
    const [channels, setChannels] = useState<Channel[]>([]);
    const [selectedChannel, setSelectedChannel] = useState<Channel | null>(null);
    const [officialData, setOfficialData] = useState<PaginatedResponse<OfficialKey>>({ items: [], total: 0, page: 1, size: 10 });

    // UI State
    const [isChannelDialogOpen, setIsChannelDialogOpen] = useState(false);
    const [editingChannel, setEditingChannel] = useState<Channel | null>(null);
    const [channelForm, setChannelForm] = useState({
        name: '',
        type: 'gemini',
        api_url: 'https://generativelanguage.googleapis.com'
    });

    const [isOfficialDialogOpen, setIsOfficialDialogOpen] = useState(false);
    const [officialForm, setOfficialForm] = useState({ key: '', is_active: true, is_batch: false });
    const [officialStatusFilter, setOfficialStatusFilter] = useState('all');
    const [selectedOfficialIds, setSelectedOfficialIds] = useState<Set<number>>(new Set());

    const { toast } = useToast();

    // Fetch Channels
    const fetchChannels = useCallback(async () => {
        const token = localStorage.getItem('token');
        try {
            const res = await axios.get<Channel[]>(`${API_BASE_URL}/channels/`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setChannels(res.data);
            // 默认选择第一个渠道
            if (res.data.length > 0 && !selectedChannel) {
                setSelectedChannel(res.data[0]);
            }
        } catch (error) {
            console.error('Failed to fetch channels', error);
            toast({ variant: 'error', title: '加载渠道失败' });
        }
    }, [selectedChannel, toast]);

    // Fetch Official Keys for selected channel
    const fetchOfficialKeys = useCallback(async (page = officialData.page, size = officialData.size) => {
        if (!selectedChannel) return;

        const token = localStorage.getItem('token');
        try {
            const res = await axios.get<PaginatedResponse<OfficialKey>>(`${API_BASE_URL}/keys/official`, {
                headers: { Authorization: `Bearer ${token}` },
                params: {
                    page,
                    size,
                    status: officialStatusFilter,
                    channel_id: selectedChannel.id
                }
            });
            setOfficialData(res.data);
            setSelectedOfficialIds(new Set());
        } catch (error) {
            console.error('Failed to fetch official keys', error);
            toast({ variant: 'error', title: '加载密钥失败' });
        }
    }, [selectedChannel, officialData.page, officialData.size, officialStatusFilter, toast]);

    useEffect(() => {
        fetchChannels();
    }, []);

    useEffect(() => {
        if (selectedChannel) {
            fetchOfficialKeys(1, 10);
        }
    }, [selectedChannel, officialStatusFilter]);

    // Channel Actions
    const handleOpenChannelDialog = (channel: Channel | null = null) => {
        if (channel) {
            setEditingChannel(channel);
            setChannelForm({
                name: channel.name,
                type: channel.type,
                api_url: channel.api_url
            });
        } else {
            setEditingChannel(null);
            setChannelForm({
                name: '',
                type: 'gemini',
                api_url: 'https://generativelanguage.googleapis.com'
            });
        }
        setIsChannelDialogOpen(true);
    };

    const handleSaveChannel = async (e: React.FormEvent) => {
        e.preventDefault();
        const token = localStorage.getItem('token');
        const headers = { Authorization: `Bearer ${token}` };

        try {
            if (editingChannel) {
                await axios.patch(`${API_BASE_URL}/channels/${editingChannel.id}`, channelForm, { headers });
                toast({ variant: 'success', title: '更新成功' });
            } else {
                await axios.post(`${API_BASE_URL}/channels/`, channelForm, { headers });
                toast({ variant: 'success', title: '创建成功' });
            }
            setIsChannelDialogOpen(false);
            fetchChannels();
        } catch (error) {
            toast({ variant: 'error', title: editingChannel ? '更新失败' : '创建失败' });
        }
    };

    const handleDeleteChannel = async (id: number) => {
        if (!await confirm({ title: "删除渠道", description: "确定要删除此渠道吗？关联的密钥将失去渠道绑定。", confirmText: "删除" })) return;
        const token = localStorage.getItem('token');
        try {
            await axios.delete(`${API_BASE_URL}/channels/${id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            fetchChannels();
            if (selectedChannel?.id === id) {
                setSelectedChannel(null);
            }
            toast({ variant: 'success', title: '删除成功' });
        } catch (error) {
            toast({ variant: 'error', title: '删除失败' });
        }
    };

    // Official Key Actions
    const handleCreateOfficial = async (e: React.FormEvent) => {
        e.preventDefault();
        const token = localStorage.getItem('token');
        const headers = { Authorization: `Bearer ${token}` };

        if (!selectedChannel) {
            toast({ variant: 'error', title: '请先选择一个渠道' });
            return;
        }

        if (officialForm.is_batch) {
            const keys = officialForm.key.split('\n').map(k => k.trim()).filter(k => k);
            if (keys.length === 0) {
                toast({ variant: 'error', title: '请输入至少一个 API Key' });
                return;
            }

            const chunkSize = 100;
            let successCount = 0;
            let failCount = 0;
            setIsOfficialDialogOpen(false);

            for (let i = 0; i < keys.length; i += chunkSize) {
                const chunk = keys.slice(i, i + chunkSize);
                const payload = {
                    keys: chunk,
                    is_active: officialForm.is_active,
                    channel_id: selectedChannel.id,
                };
                try {
                    const response = await axios.post(`${API_BASE_URL}/keys/official/batch`, payload, { headers });
                    successCount += response.data.success_count || chunk.length;
                    failCount += response.data.fail_count || 0;
                } catch (error) {
                    failCount += chunk.length;
                    console.error('批量添加失败', error);
                }
                toast({
                    title: `处理中... (${i + chunk.length}/${keys.length})`,
                    description: `成功: ${successCount}, 失败: ${failCount}`,
                });
            }

            toast({
                variant: 'success',
                title: '批量添加完成',
                description: `总计: 成功 ${successCount} 个, 失败 ${failCount} 个`,
            });
            setOfficialForm({ key: '', is_active: true, is_batch: false });
            fetchOfficialKeys(1, officialData.size);

        } else {
            try {
                await axios.post(`${API_BASE_URL}/keys/official`, {
                    key: officialForm.key,
                    is_active: officialForm.is_active,
                    channel_id: selectedChannel.id
                }, { headers });
                setIsOfficialDialogOpen(false);
                setOfficialForm({ key: '', is_active: true, is_batch: false });
                fetchOfficialKeys(officialData.page, officialData.size);
                toast({ variant: 'success', title: '添加成功' });
            } catch (error) {
                toast({ variant: 'error', title: '添加失败' });
            }
        }
    };

    const handleToggleOfficialActive = async (id: number, isActive: boolean) => {
        const token = localStorage.getItem('token');
        try {
            await axios.patch(`${API_BASE_URL}/keys/official/${id}`,
                { is_active: isActive },
                { headers: { Authorization: `Bearer ${token}` } }
            );
            fetchOfficialKeys(officialData.page, officialData.size);
            toast({ variant: 'success', title: '状态已更新' });
        } catch (error) {
            toast({ variant: 'error', title: '更新失败' });
        }
    };

    const handleDeleteOfficial = async (id: number) => {
        if (!await confirm({ title: "删除密钥", description: `确定要删除此官方密钥吗？`, confirmText: "删除" })) return;
        const token = localStorage.getItem('token');
        try {
            await axios.delete(`${API_BASE_URL}/keys/official/${id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            fetchOfficialKeys(officialData.page, officialData.size);
            toast({ variant: 'success', title: '删除成功' });
        } catch (error) {
            toast({ variant: 'error', title: '删除失败' });
        }
    };

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        toast({ title: '已复制到剪贴板' });
    };

    // Selection Logic
    const handleSelectAll = () => {
        if (selectedOfficialIds.size === officialData.items.length) {
            setSelectedOfficialIds(new Set());
        } else {
            setSelectedOfficialIds(new Set(officialData.items.map(k => k.id)));
        }
    };

    const handleSelectInverse = () => {
        const newSet = new Set<number>();
        officialData.items.forEach(k => {
            if (!selectedOfficialIds.has(k.id)) newSet.add(k.id);
        });
        setSelectedOfficialIds(newSet);
    };

    const handleSelectOne = (id: number) => {
        const newSet = new Set(selectedOfficialIds);
        if (newSet.has(id)) newSet.delete(id);
        else newSet.add(id);
        setSelectedOfficialIds(newSet);
    };

    // Batch Operations
    const handleBatchAction = async (action: 'enable' | 'disable' | 'delete') => {
        const ids = Array.from(selectedOfficialIds);
        if (ids.length === 0) return;

        if (action === 'delete') {
            if (!await confirm({
                title: "批量删除",
                description: `确定要删除选中的 ${ids.length} 个密钥吗？此操作不可恢复。`,
                confirmText: "删除"
            })) return;
        }

        const token = localStorage.getItem('token');
        const headers = { Authorization: `Bearer ${token}` };
        let successCount = 0;
        let failCount = 0;

        const promises = ids.map(id => {
            const url = `${API_BASE_URL}/keys/official/${id}`;
            if (action === 'delete') {
                return axios.delete(url, { headers });
            } else {
                return axios.patch(url, { is_active: action === 'enable' }, { headers });
            }
        });

        const results = await Promise.allSettled(promises);

        results.forEach(res => {
            if (res.status === 'fulfilled') successCount++;
            else failCount++;
        });

        if (successCount > 0) {
            toast({ variant: 'success', title: `成功${action === 'delete' ? '删除' : (action === 'enable' ? '启用' : '禁用')} ${successCount} 个密钥` });
        }
        if (failCount > 0) {
            toast({ variant: 'error', title: `${failCount} 个密钥操作失败` });
        }

        fetchOfficialKeys(officialData.page, officialData.size);
        setSelectedOfficialIds(new Set());
    };

    const handleChannelTypeChange = (type: string) => {
        const channelType = DEFAULT_CHANNEL_TYPES.find(t => t.value === type);
        setChannelForm({
            ...channelForm,
            type,
            api_url: channelType?.defaultUrl || ''
        });
    };

    return (
        <div className="flex h-full gap-4 overflow-hidden">
            {/* Left Sidebar - Channel List */}
            <div className="w-64 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold">渠道列表</h2>
                    <Button size="sm" onClick={() => handleOpenChannelDialog()}>
                        <Plus className="w-4 h-4" />
                    </Button>
                </div>

                <Card className="flex-1 overflow-hidden">
                    <CardContent className="p-2 h-full overflow-y-auto">
                        <div className="space-y-1">
                            {channels.map((channel) => (
                                <div
                                    key={channel.id}
                                    className={cn(
                                        "flex items-center justify-between p-3 rounded-md cursor-pointer transition-colors group",
                                        selectedChannel?.id === channel.id
                                            ? "bg-primary text-primary-foreground"
                                            : "hover:bg-accent"
                                    )}
                                    onClick={() => setSelectedChannel(channel)}
                                >
                                    <div className="flex-1 min-w-0">
                                        <div className="font-medium truncate">{channel.name}</div>
                                        <div className={cn(
                                            "text-xs truncate",
                                            selectedChannel?.id === channel.id
                                                ? "text-primary-foreground/80"
                                                : "text-muted-foreground"
                                        )}>
                                            密钥: {channel.active_keys} / {channel.total_keys}
                                        </div>
                                    </div>
                                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleOpenChannelDialog(channel);
                                            }}
                                        >
                                            <Settings className="w-3 h-3" />
                                        </Button>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6 text-destructive hover:text-destructive"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteChannel(channel.id);
                                            }}
                                        >
                                            <Trash2 className="w-3 h-3" />
                                        </Button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Right Content - Official Keys */}
            <div className="flex-1 flex flex-col gap-4 min-w-0 overflow-hidden">
                <div className="flex flex-col gap-2">
                    <h1 className="text-3xl font-bold tracking-tight">
                        {selectedChannel ? `${selectedChannel.name} 渠道密钥` : '选择渠道'}
                    </h1>
                    {selectedChannel && (
                        <p className="text-sm text-muted-foreground">
                            API 地址: {selectedChannel.api_url}
                        </p>
                    )}
                </div>

                {selectedChannel ? (
                    <>
                        <div className="flex justify-between items-center">
                            <div className="flex gap-4">
                                <div className="flex items-center gap-2 border px-3 py-1 rounded-md text-sm h-10 bg-card">
                                    <span className="text-muted-foreground">总密钥数</span>
                                    <span className="font-bold">{officialData.total}</span>
                                </div>
                                <Select value={officialStatusFilter} onValueChange={setOfficialStatusFilter}>
                                    <SelectTrigger className="w-[120px]">
                                        <SelectValue placeholder="状态筛选" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="all">全部</SelectItem>
                                        <SelectItem value="normal">正常</SelectItem>
                                        <SelectItem value="abnormal">异常</SelectItem>
                                        <SelectItem value="manually_disabled">手动禁用</SelectItem>
                                        <SelectItem value="auto_disabled">自动禁用</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="flex gap-2">
                                <Button onClick={() => fetchOfficialKeys(officialData.page, officialData.size)} variant="outline" className="h-10">
                                    <RefreshCw className="w-4 h-4 mr-2" />
                                    刷新
                                </Button>
                                <Button onClick={() => setIsOfficialDialogOpen(true)} className="h-10">
                                    <Plus className="w-4 h-4 mr-2" />
                                    添加密钥
                                </Button>
                            </div>
                        </div>

                        <Card className="flex-1 overflow-hidden">
                            <CardContent className="p-0 h-full flex flex-col">
                                <div className="flex-1 overflow-x-auto">
                                    <table className="w-full caption-bottom text-sm">
                                        <thead className="[&_tr]:border-b">
                                            <tr className="border-b transition-colors hover:bg-muted/50">
                                                <th className="h-12 w-[50px] px-4 align-middle">
                                                    <Checkbox
                                                        checked={officialData.items.length > 0 && selectedOfficialIds.size === officialData.items.length}
                                                        onCheckedChange={handleSelectAll}
                                                    />
                                                </th>
                                                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">密钥</th>
                                                <th className="h-12 px-4 text-center align-middle font-medium text-muted-foreground">状态</th>
                                                <th className="h-12 px-4 text-center align-middle font-medium text-muted-foreground">状态码</th>
                                                <th className="h-12 px-4 text-center align-middle font-medium text-muted-foreground">请求/错误</th>
                                                <th className="h-12 px-4 text-center align-middle font-medium text-muted-foreground">令牌 (输入/输出)</th>
                                                <th className="h-12 px-4 text-center align-middle font-medium text-muted-foreground">创建时间</th>
                                                <th className="h-12 px-4 text-right align-middle font-medium text-muted-foreground"></th>
                                            </tr>
                                        </thead>
                                        <tbody className="[&_tr:last-child]:border-0">
                                            {officialData.items.length === 0 ? (
                                                <tr>
                                                    <td colSpan={8} className="p-8 text-center text-muted-foreground">
                                                        <div className="flex flex-col items-center justify-center gap-2">
                                                            <AlertCircle className="h-8 w-8 opacity-50" />
                                                            <p>暂无密钥</p>
                                                        </div>
                                                    </td>
                                                </tr>
                                            ) : (
                                                officialData.items.map((key) => {
                                                    let healthColor = 'text-green-600';
                                                    let healthLabel = '正常';

                                                    if (key.is_active) {
                                                        if (key.last_status === '200' || key.last_status === 'active') {
                                                            healthColor = 'text-green-600';
                                                            healthLabel = '正常';
                                                        } else {
                                                            healthColor = 'text-red-600';
                                                            healthLabel = '异常';
                                                        }
                                                    } else {
                                                        if (key.last_status === 'auto_disabled') {
                                                            healthColor = 'text-yellow-600';
                                                            healthLabel = '自动禁用';
                                                        } else {
                                                            healthColor = 'text-gray-500';
                                                            healthLabel = '手动禁用';
                                                        }
                                                    }

                                                    return (
                                                        <tr key={key.id} className="border-b transition-colors hover:bg-muted/50 group">
                                                            <td className="p-4 align-middle">
                                                                <Checkbox
                                                                    checked={selectedOfficialIds.has(key.id)}
                                                                    onCheckedChange={() => handleSelectOne(key.id)}
                                                                />
                                                            </td>
                                                            <td className="p-4 align-middle">
                                                                <div className="flex items-center gap-2">
                                                                    <MaskedKey apiKey={key.key} />
                                                                    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => copyToClipboard(key.key)}>
                                                                        <Copy className="w-3 h-3" />
                                                                    </Button>
                                                                </div>
                                                            </td>
                                                            <td className="p-4 align-middle">
                                                                <div className="flex items-center justify-center gap-2">
                                                                    <Activity className={cn("w-4 h-4", healthColor)} />
                                                                    <span className={healthColor}>{healthLabel}</span>
                                                                </div>
                                                            </td>
                                                            <td className="p-4 align-middle text-center">
                                                                <span className="font-mono text-xs">{key.last_status_code || '-'}</span>
                                                            </td>
                                                            <td className="p-4 align-middle text-center">
                                                                <div className="flex flex-col text-sm">
                                                                    <span>{key.usage_count}</span>
                                                                    <span className="text-xs text-muted-foreground">{key.error_count} 错误</span>
                                                                </div>
                                                            </td>
                                                            <td className="p-4 align-middle text-center font-mono text-sm">
                                                                {key.input_tokens.toLocaleString()} / {key.output_tokens.toLocaleString()}
                                                            </td>
                                                            <td className="p-4 align-middle text-center text-sm text-muted-foreground">
                                                                {new Date(key.created_at).toLocaleDateString()}
                                                            </td>
                                                            <td className="p-4 align-middle text-right">
                                                                <div className="flex justify-end gap-2 items-center">
                                                                    <Switch
                                                                        checked={key.is_active}
                                                                        onCheckedChange={(checked) => handleToggleOfficialActive(key.id, checked)}
                                                                    />
                                                                    <Button variant="ghost" size="icon" className="text-destructive opacity-0 group-hover:opacity-100 transition-opacity" onClick={() => handleDeleteOfficial(key.id)}>
                                                                        <Trash2 className="w-4 h-4" />
                                                                    </Button>
                                                                </div>
                                                            </td>
                                                        </tr>
                                                    );
                                                })
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </CardContent>
                        </Card>

                        <div className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                                <Button variant="outline" size="sm" onClick={handleSelectInverse}>
                                    反选
                                </Button>
                                {selectedOfficialIds.size > 0 ? (
                                    <>
                                        <div className="h-4 w-[1px] bg-border mx-1" />
                                        <span className="text-sm text-muted-foreground">已选择 {selectedOfficialIds.size} 项</span>
                                        <div className="h-4 w-[1px] bg-border mx-1" />
                                        <Button variant="ghost" size="sm" onClick={() => handleBatchAction('enable')}>
                                            启用
                                        </Button>
                                        <Button variant="ghost" size="sm" onClick={() => handleBatchAction('disable')}>
                                            禁用
                                        </Button>
                                        <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => handleBatchAction('delete')}>
                                            删除
                                        </Button>
                                    </>
                                ) : null}
                            </div>
                            <Pagination
                                currentPage={officialData.page}
                                totalPages={Math.ceil(officialData.total / officialData.size)}
                                pageSize={officialData.size}
                                totalItems={officialData.total}
                                onPageChange={(page) => fetchOfficialKeys(page, officialData.size)}
                                onPageSizeChange={(size) => fetchOfficialKeys(1, size)}
                            />
                        </div>
                    </>
                ) : (
                    <Card className="flex-1">
                        <CardContent className="flex items-center justify-center h-full">
                            <div className="text-center text-muted-foreground">
                                <p className="text-lg">请从左侧选择一个渠道</p>
                            </div>
                        </CardContent>
                    </Card>
                )}
            </div>

            {/* Channel Dialog */}
            <Dialog open={isChannelDialogOpen} onOpenChange={setIsChannelDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>{editingChannel ? '编辑渠道' : '添加渠道'}</DialogTitle>
                    </DialogHeader>
                    <form onSubmit={handleSaveChannel} className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="channel-name">渠道名称</Label>
                            <Input
                                id="channel-name"
                                value={channelForm.name}
                                onChange={(e) => setChannelForm({ ...channelForm, name: e.target.value })}
                                placeholder="My Gemini Channel"
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label>渠道类型</Label>
                            <Select
                                value={channelForm.type}
                                onValueChange={handleChannelTypeChange}
                            >
                                <SelectTrigger>
                                    <SelectValue placeholder="选择类型..." />
                                </SelectTrigger>
                                <SelectContent>
                                    {DEFAULT_CHANNEL_TYPES.map(type => (
                                        <SelectItem key={type.value} value={type.value}>
                                            {type.label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="channel-url">API 地址</Label>
                            <Input
                                id="channel-url"
                                value={channelForm.api_url}
                                onChange={(e) => setChannelForm({ ...channelForm, api_url: e.target.value })}
                                placeholder="https://..."
                                required
                            />
                        </div>
                        <DialogFooter>
                            <Button type="submit">{editingChannel ? '保存' : '创建'}</Button>
                        </DialogFooter>
                    </form>
                </DialogContent>
            </Dialog>

            {/* Official Key Dialog */}
            <Dialog open={isOfficialDialogOpen} onOpenChange={setIsOfficialDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>添加官方密钥到 {selectedChannel?.name}</DialogTitle>
                    </DialogHeader>
                    <form onSubmit={handleCreateOfficial} className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="key">{officialForm.is_batch ? 'API Keys (一行一个)' : 'API Key'}</Label>
                            {officialForm.is_batch ? (
                                <Textarea
                                    id="key-batch"
                                    value={officialForm.key}
                                    onChange={(e) => setOfficialForm({ ...officialForm, key: e.target.value })}
                                    placeholder="AIza...\nAIza...\nAIza..."
                                    required
                                    className="min-h-[120px] font-mono"
                                />
                            ) : (
                                <Input
                                    id="key"
                                    type="text"
                                    value={officialForm.key}
                                    onChange={(e) => setOfficialForm({ ...officialForm, key: e.target.value })}
                                    placeholder="AIza..."
                                    required
                                />
                            )}
                        </div>
                        <div className="flex items-center justify-between">
                            <Label htmlFor="active">启用</Label>
                            <Switch
                                id="active"
                                checked={officialForm.is_active}
                                onCheckedChange={(checked) => setOfficialForm({ ...officialForm, is_active: checked })}
                            />
                        </div>
                        <div className="flex items-center justify-between">
                            <Label htmlFor="batch-mode">批量模式</Label>
                            <Switch
                                id="batch-mode"
                                checked={officialForm.is_batch}
                                onCheckedChange={(checked) => setOfficialForm({ ...officialForm, is_batch: checked, key: '' })}
                            />
                        </div>
                        <DialogFooter>
                            <Button type="submit">添加</Button>
                        </DialogFooter>
                    </form>
                </DialogContent>
            </Dialog>
        </div>
    );
}
