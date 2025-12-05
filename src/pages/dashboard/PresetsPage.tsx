import { useState, useEffect } from 'react';
import { PresetItemEditor } from '@/components/preset/PresetItemEditor';
import { PresetRegexPage } from '@/components/preset/PresetRegexPage';
import { useToast } from '@/hooks/useToast';
import { presetService, type Preset } from '@/services/presetService';
import { presetRegexService } from '@/services/presetRegexService';
import { format } from 'date-fns';
import { toZonedTime } from 'date-fns-tz';
import { exportToJSON, importFromJSON } from '@/utils/exportImport';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import {
    Plus,
    Copy,
    Trash2,
    Download,
    Upload,
    Pencil,
    ChevronDown,
} from 'lucide-react';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { confirm } from '@/components/ui/ConfirmDialog';

export default function PresetsPage() {
    const [presets, setPresets] = useState<Preset[]>([]);
    const [selectedPreset, setSelectedPreset] = useState<Preset | null>(null);
    const [loading, setLoading] = useState(true);

    // Rename Dialog State
    const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false);
    const [renameName, setRenameName] = useState('');

    // Select Open State
    const [isSelectOpen, setIsSelectOpen] = useState(false);

    const { toast } = useToast();

    // 加载预设列表
    const fetchPresets = async (newlySelectedId: number | null = null) => {
        try {
            setLoading(true);
            const data = await presetService.getPresets();
            const sortedData = data.sort((a, b) => a.sort_order - b.sort_order);
            setPresets(sortedData);

            let presetToSelect: Preset | null = null;
            if (newlySelectedId) {
                presetToSelect = sortedData.find(p => p.id === newlySelectedId) || null;
            }

            if (!presetToSelect && selectedPreset) {
                // Try to find the previously selected preset in the new list
                const updated = sortedData.find((p) => p.id === selectedPreset.id);
                if (updated) {
                    presetToSelect = updated;
                }
            }

            // Fallback to the first preset if no selection is determined yet
            if (!presetToSelect && sortedData.length > 0) {
                presetToSelect = sortedData[0];
            }

            setSelectedPreset(presetToSelect);
        } catch (error) {
            toast({
                variant: 'error',
                title: '加载失败',
                description: '无法加载预设列表',
            });
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPresets();
    }, []);

    // 创建新预设
    const handleCreatePreset = async () => {
        try {
            const defaultContent = {
                preset: [
                    {
                        name: '默认条目',
                        role: 'system',
                        type: 'normal',
                        content: '你是一个得力助手。',
                        enabled: true,
                    },
                ],
                regex: [],
            };

            const newPresetData = await presetService.createPreset({
                name: '新建预设',
                is_active: true,
                sort_order: presets.length,
                content: JSON.stringify(defaultContent),
            });

            // The backend now returns the full preset with items if they are created from content
            // We need to parse the content string back to an object for the frontend state
            if (newPresetData.content && typeof newPresetData.content === 'string') {
                try {
                    JSON.parse(newPresetData.content);
                } catch (e) {
                    console.error("Failed to parse content from newly created preset", e);
                }
            }

            // The API should ideally return the created items directly.
            // For now, let's just refetch to get the complete data.
            await fetchPresets(newPresetData.id);

            toast({
                variant: 'success',
                title: '创建成功',
                description: '新预设已创建并包含默认条目',
            });
        } catch (error) {
            toast({
                variant: 'error',
                title: '创建失败',
                description: '无法创建新预设',
            });
        }
    };


    // 打开重命名弹窗
    const openRenameDialog = () => {
        if (!selectedPreset) return;
        setRenameName(selectedPreset.name);
        setIsRenameDialogOpen(true);
    };

    // 执行重命名
    const handleRename = async () => {
        if (!selectedPreset || !renameName.trim()) return;
        try {
            await presetService.updatePreset(selectedPreset.id, {
                name: renameName,
                is_active: selectedPreset.is_active,
                sort_order: selectedPreset.sort_order,
            });

            const updatedPreset = { ...selectedPreset, name: renameName };
            setPresets(presets.map((p) => (p.id === selectedPreset.id ? updatedPreset : p)));
            setSelectedPreset(updatedPreset);
            setIsRenameDialogOpen(false);
            toast({
                variant: 'success',
                title: '重命名成功',
            });
        } catch (error) {
            toast({
                variant: 'error',
                title: '重命名失败',
            });
        }
    };

    // 删除预设
    const handleDelete = async () => {
        if (!selectedPreset) return;
        if (!await confirm({ title: "删除预设", description: `确定要删除预设 "${selectedPreset.name}" 吗？`, confirmText: "删除", cancelText: "取消" })) return;

        try {
            await presetService.deletePreset(selectedPreset.id);
            const newPresets = presets.filter((p) => p.id !== selectedPreset.id);
            setPresets(newPresets);

            if (newPresets.length > 0) {
                setSelectedPreset(newPresets[0]);
            } else {
                setSelectedPreset(null);
            }

            toast({
                variant: 'success',
                title: '删除成功',
            });
        } catch (error) {
            toast({
                variant: 'error',
                title: '删除失败',
            });
        }
    };

    // 复制预设
    const handleDuplicate = async () => {
        if (!selectedPreset) return;
        try {
            const newPreset = await presetService.createPreset({
                name: `${selectedPreset.name} (副本)`,
                is_active: selectedPreset.is_active,
                sort_order: presets.length,
                content: selectedPreset.content || '',
            });
            setPresets([...presets, newPreset]);
            setSelectedPreset(newPreset);
            toast({
                variant: 'success',
                title: '复制成功',
            });
        } catch (error) {
            toast({
                variant: 'error',
                title: '复制失败',
            });
        }
    };

    // 导出预设（仅导出选中的）
    const handleExport = async () => {
        if (!selectedPreset) {
            toast({ variant: 'error', title: '请先选择一个预设' });
            return;
        }

        try {
            // 1. 获取预设内容
            // The preset items are now directly available on the selectedPreset object.
            const presetItems = selectedPreset.items || [];

            // 2. 获取关联的正则规则
            const regexRules = await presetRegexService.getPresetRegexRules(selectedPreset.id);

            // 3. 格式化正则规则用于导出
            const formattedRegex = regexRules.map(r => ({
                name: r.name,
                creator_username: r.creator_username,
                created_at: r.created_at ? format(toZonedTime(new Date(r.created_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss') : '',
                updated_at: r.updated_at ? format(toZonedTime(new Date(r.updated_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss') : '',
                enabled: r.is_active, // 注意字段名转换
                content: {
                    type: r.type,
                    pattern: r.pattern,
                    replacement: r.replacement,
                }
            }));

            // 4. 组合成新的导出格式
            const exportData = {
                name: selectedPreset.name,
                type: 'preset',
                creator_username: (selectedPreset as any).creator_username || 'unknown', // creator_username not in interface
                created_at: format(toZonedTime(new Date(selectedPreset.created_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
                updated_at: format(toZonedTime(new Date(selectedPreset.updated_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
                enabled: selectedPreset.is_active,
                content: {
                    preset: presetItems.map(item => ({
                        name: item.name,
                        creator_username: item.creator_username || 'unknown',
                        created_at: format(toZonedTime(new Date(item.created_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
                        updated_at: format(toZonedTime(new Date(item.updated_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
                        enabled: item.enabled,
                        role: item.role,
                        type: item.type,
                        content: item.content,
                    })),
                    regex: formattedRegex,
                }
            };

            exportToJSON(exportData, `any-api-preset-${selectedPreset.name}`);
            toast({
                variant: 'success',
                title: '导出成功',
            });

        } catch (error) {
            toast({
                variant: 'error',
                title: '导出失败',
                description: '无法获取关联的正则数据',
            });
        }
    };

    // 导入预设
    const handleImport = async () => {
        try {
            const importedRaw = await importFromJSON<string>(true) as string; // Import as raw text
            const importedData = JSON.parse(importedRaw);


            // Case 1: Import a full preset
            if (importedData.type === 'preset') {
                const newPreset = await presetService.createPreset({
                    name: importedData.name,
                    is_active: importedData.enabled,
                    sort_order: presets.length,
                    content: importedRaw, // Pass the raw JSON content
                });

                if (importedData.content && importedData.content.preset) {
                    for (const item of importedData.content.preset) {
                        await presetService.createPresetItem(newPreset.id, {
                            name: item.name,
                            role: item.role,
                            type: item.type,
                            content: item.content,
                            enabled: item.enabled,
                            sort_order: importedData.content.preset.indexOf(item),
                        });
                    }
                }

                if (importedData.content && importedData.content.regex) {
                    for (const regexRule of importedData.content.regex) {
                        await presetRegexService.createPresetRegexRule(newPreset.id, {
                            name: regexRule.name,
                            pattern: regexRule.content.pattern,
                            replacement: regexRule.content.replacement,
                            type: regexRule.content.type,
                            is_active: regexRule.enabled,
                            sort_order: importedData.content.regex.indexOf(regexRule),
                        });
                    }
                }
                toast({
                    variant: 'success',
                    title: '导入成功',
                    description: `成功导入预设 "${importedData.name}"`,
                });

                // Refresh presets list and select the newly imported one
                await fetchPresets(newPreset.id);
            }
            // Case 2: Import regex rules into the selected preset
            else {
                const rulesToImport = Array.isArray(importedData) ? importedData : [importedData];
                if (rulesToImport.every(r => r.type === 'regex' && r.content)) {
                    // This is a regex-only file. Guide the user to the correct import location.
                    toast({
                        variant: 'info',
                        title: '检测到正则规则文件',
                        description: '请在预设的“正则管理”选项卡中导入此文件。',
                        duration: 5000,
                    });
                } else {
                    throw new Error('文件格式不兼容。请选择一个预设文件。');
                }
            }
        } catch (error) {
            console.error("Import failed:", error);
            toast({
                variant: 'error',
                title: '导入失败',
                description: error instanceof Error ? error.message : '未知错误',
            });
        }
    };


    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">加载中...</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-[calc(100vh-8rem)] gap-4">
            {/* 功能工具栏 */}
            <div className="flex items-center gap-4 px-4 py-2 bg-card border rounded-lg shadow-sm">
                {/* 预设选择下拉框 */}
                <div className="flex-1 max-w-xs relative">
                    <Select
                        open={isSelectOpen}
                        onOpenChange={setIsSelectOpen}
                        value={selectedPreset?.id.toString()}
                        onValueChange={(val) => {
                            const preset = presets.find(p => p.id === parseInt(val));
                            if (preset) setSelectedPreset(preset);
                        }}
                    >
                        <SelectTrigger className={selectedPreset ? "pr-24 [&>svg]:hidden" : "pr-8"}>
                            <SelectValue placeholder="选择预设..." />
                        </SelectTrigger>
                        <SelectContent>
                            {presets.map((preset) => (
                                <div key={preset.id} className="relative group">
                                    <SelectItem value={preset.id.toString()} className="pr-16">
                                        {preset.name}
                                    </SelectItem>
                                    {/* 列表项右侧的重命名和删除按钮 */}
                                    <div
                                        className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity pointer-events-auto"
                                        onClick={(e) => e.stopPropagation()}
                                        onPointerDown={(e) => e.stopPropagation()}
                                    >
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                e.preventDefault();
                                                setSelectedPreset(preset);
                                                setRenameName(preset.name);
                                                setIsRenameDialogOpen(true);
                                            }}
                                            onPointerDown={(e) => e.stopPropagation()}
                                            title="重命名"
                                        >
                                            <Pencil className="w-3 h-3" />
                                        </Button>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-6 w-6 text-destructive hover:text-destructive"
                                            onClick={async (e) => {
                                                e.stopPropagation();
                                                e.preventDefault();
                                                if (await confirm({ title: "删除预设", description: `确定要删除预设 "${preset.name}" 吗？`, confirmText: "删除", cancelText: "取消" })) {
                                                    try {
                                                        await presetService.deletePreset(preset.id);
                                                        const newPresets = presets.filter((p) => p.id !== preset.id);
                                                        setPresets(newPresets);
                                                        if (selectedPreset?.id === preset.id) {
                                                            setSelectedPreset(newPresets.length > 0 ? newPresets[0] : null);
                                                        }
                                                        toast({ variant: 'success', title: '删除成功' });
                                                    } catch (error) {
                                                        toast({ variant: 'error', title: '删除失败' });
                                                    }
                                                }
                                            }}
                                            onPointerDown={(e) => e.stopPropagation()}
                                            title="删除"
                                        >
                                            <Trash2 className="w-3 h-3" />
                                        </Button>
                                    </div>
                                </div>
                            ))}
                        </SelectContent>
                    </Select>

                    {/* 下拉框内的重命名和删除按钮 - 移到 Select 外部并绝对定位 */}
                    {selectedPreset && (
                        <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 z-10 pointer-events-none">
                            <ChevronDown
                                className={cn(
                                    "h-4 w-4 opacity-50 transition-transform duration-200",
                                    isSelectOpen && "rotate-180"
                                )}
                            />
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 pointer-events-auto"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    openRenameDialog();
                                }}
                                title="重命名选中的预设"
                            >
                                <Pencil className="w-3 h-3" />
                            </Button>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-destructive hover:text-destructive pointer-events-auto"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    handleDelete();
                                }}
                                title="删除选中的预设"
                            >
                                <Trash2 className="w-3 h-3" />
                            </Button>
                        </div>
                    )}
                </div>

                {/* 操作按钮组 */}
                <div className="flex items-center gap-1">
                    <Button variant="ghost" size="icon" onClick={handleCreatePreset} title="新建预设">
                        <Plus className="w-4 h-4" />
                    </Button>
                    <Button variant="ghost" size="icon" onClick={handleDuplicate} disabled={!selectedPreset} title="复制预设">
                        <Copy className="w-4 h-4" />
                    </Button>
                    <div className="w-px h-4 bg-border mx-1" />
                    <Button variant="ghost" size="icon" onClick={handleExport} disabled={!selectedPreset} title="导出选中预设">
                        <Download className="w-4 h-4" />
                    </Button>
                    <Button variant="ghost" size="icon" onClick={handleImport} title="导入预设">
                        <Upload className="w-4 h-4" />
                    </Button>
                </div>
            </div>

            {/* 主内容区域 */}
            <div className="flex-1 border rounded-lg bg-card overflow-hidden">
                {!selectedPreset ? (
                    <div className="flex items-center justify-center h-full text-muted-foreground">
                        <p>请选择或创建一个预设</p>
                    </div>
                ) : (
                    <Tabs defaultValue="items" className="h-full flex flex-col">
                        <TabsList className="w-full rounded-none border-b bg-muted/50 justify-start px-4">
                            <TabsTrigger value="items">预设管理</TabsTrigger>
                            <TabsTrigger value="regex">正则管理</TabsTrigger>
                        </TabsList>
                        <TabsContent value="items" className="flex-1 m-0 overflow-hidden">
                            <PresetItemEditor
                                preset={selectedPreset}
                                onItemsChange={fetchPresets}
                            />
                        </TabsContent>
                        <TabsContent value="regex" className="flex-1 m-0 overflow-hidden">
                            <PresetRegexPage presetId={selectedPreset.id} />
                        </TabsContent>
                    </Tabs>
                )}
            </div>

            {/* 重命名弹窗 */}
            <Dialog open={isRenameDialogOpen} onOpenChange={setIsRenameDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>重命名预设</DialogTitle>
                        <DialogDescription>
                            请输入新的预设名称
                        </DialogDescription>
                    </DialogHeader>
                    <div className="py-4">
                        <Input
                            value={renameName}
                            onChange={(e) => setRenameName(e.target.value)}
                            placeholder="预设名称"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter') handleRename();
                            }}
                        />
                    </div>
                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsRenameDialogOpen(false)}>取消</Button>
                        <Button onClick={handleRename}>确定</Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </div>
    );
}
