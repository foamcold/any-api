import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '@/utils/api';
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors, DragEndEvent } from '@dnd-kit/core';
import { arrayMove, SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Plus, Pencil, Trash2, GripVertical, Download, Upload, FileDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
    DialogFooter,
    DialogDescription,
} from '@/components/ui/dialog';
import { useToast } from '@/hooks/useToast';
import { format } from 'date-fns';
import { toZonedTime } from 'date-fns-tz';
import { exportToJSON, importFromJSON } from '@/utils/exportImport';
import { cn } from '@/lib/utils';
import { confirm } from '@/components/ui/ConfirmDialog';

interface RegexRule {
    id: number;
    name: string;
    pattern: string;
    replacement: string;
    type: 'pre' | 'post';
    is_active: boolean;
    sort_order: number;
    created_at: string;
    updated_at: string;
}

interface SortableRuleItemProps {
    rule: RegexRule;
    onEdit: (rule: RegexRule) => void;
    onDelete: (id: number) => void;
    onExportSingle: (rule: RegexRule) => void;
    onToggle: (id: number, active: boolean) => void;
}

function SortableRuleItem({ rule, onEdit, onDelete, onExportSingle, onToggle }: SortableRuleItemProps) {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging,
    } = useSortable({ id: rule.id });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        opacity: isDragging ? 0.5 : 1,
    };

    return (
        <div
            ref={setNodeRef}
            style={style}
            className={cn(
                "p-4 flex items-center gap-4 hover:bg-accent/50 transition-colors border-b last:border-b-0",
                isDragging && "shadow-lg ring-2 ring-primary"
            )}
        >
            <div
                {...attributes}
                {...listeners}
                className="cursor-grab active:cursor-grabbing opacity-50 hover:opacity-100"
            >
                <GripVertical className="w-5 h-5 text-muted-foreground" />
            </div>
            <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium">{rule.name}</span>
                    <span className={cn(
                        "text-xs px-2 py-0.5 rounded font-medium",
                        rule.type === 'pre'
                            ? "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-100"
                            : "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-100"
                    )}>
                        {rule.type === 'pre' ? '预处理' : '后处理'}
                    </span>
                    {!rule.is_active && (
                        <span className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded">未启用</span>
                    )}
                </div>
                <div className="text-sm font-mono text-muted-foreground">
                    s/{rule.pattern}/{rule.replacement}/g
                </div>
            </div>
            <div className="flex items-center gap-2">
                <Switch
                    checked={rule.is_active}
                    onCheckedChange={(checked) => onToggle(rule.id, checked)}
                    onClick={(e) => e.stopPropagation()}
                />
                <Button variant="ghost" size="icon" onClick={() => onExportSingle(rule)} title="导出此规则">
                    <FileDown className="w-4 h-4" />
                </Button>
                <Button variant="ghost" size="icon" onClick={() => onEdit(rule)}>
                    <Pencil className="w-4 h-4" />
                </Button>
                <Button variant="ghost" size="icon" className="text-destructive" onClick={() => onDelete(rule.id)}>
                    <Trash2 className="w-4 h-4" />
                </Button>
            </div>
        </div>
    );
}

export default function RegexPage() {
    const [rules, setRules] = useState<RegexRule[]>([]);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [editingRule, setEditingRule] = useState<RegexRule | null>(null);
    const [formData, setFormData] = useState({
        name: '',
        pattern: '',
        replacement: '',
        type: 'pre' as 'pre' | 'post',
        is_active: true
    });
    const { toast } = useToast();

    const sensors = useSensors(
        useSensor(PointerSensor),
        useSensor(KeyboardSensor, {
            coordinateGetter: sortableKeyboardCoordinates,
        })
    );

    const fetchRules = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get<RegexRule[]>(`${API_BASE_URL}/regex/`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setRules(response.data.sort((a, b) => a.sort_order - b.sort_order));
        } catch (error) {
            toast({
                variant: 'error',
                title: '加载失败',
                description: '无法加载正则',
            });
        }
    };

    useEffect(() => {
        fetchRules();
    }, []);

    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;

        if (over && active.id !== over.id) {
            const oldIndex = rules.findIndex((r) => r.id === active.id);
            const newIndex = rules.findIndex((r) => r.id === over.id);

            const newRules = arrayMove(rules, oldIndex, newIndex).map((r, index) => ({
                ...r,
                sort_order: index,
            }));

            setRules(newRules);
            updateRulesOrder(newRules);
        }
    };

    const updateRulesOrder = async (updatedRules: RegexRule[]) => {
        const token = localStorage.getItem('token');
        try {
            await Promise.all(
                updatedRules.map((rule) =>
                    axios.put(`${API_BASE_URL}/regex/${rule.id}`, {
                        name: rule.name,
                        pattern: rule.pattern,
                        replacement: rule.replacement,
                        type: rule.type,
                        is_active: rule.is_active,
                        sort_order: rule.sort_order,
                    }, {
                        headers: { Authorization: `Bearer ${token}` }
                    })
                )
            );

        } catch (error) {
            toast({
                variant: 'error',
                title: '更新失败',
                description: '无法更新排序',
            });
        }
    };

    const handleToggleRule = async (id: number, active: boolean) => {
        const rule = rules.find(r => r.id === id);
        if (!rule) return;

        const token = localStorage.getItem('token');
        try {
            await axios.put(`${API_BASE_URL}/regex/${id}`, {
                name: rule.name,
                pattern: rule.pattern,
                replacement: rule.replacement,
                type: rule.type,
                is_active: active,
                sort_order: rule.sort_order,
            }, {
                headers: { Authorization: `Bearer ${token}` }
            });

            setRules(rules.map(r => r.id === id ? { ...r, is_active: active } : r));
            toast({ variant: 'success', title: active ? '已启用' : '已禁用' });
        } catch (error) {
            toast({ variant: 'error', title: '更新状态失败' });
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const token = localStorage.getItem('token');
        try {
            if (editingRule) {
                await axios.put(`${API_BASE_URL}/regex/${editingRule.id}`,
                    { ...formData, sort_order: editingRule.sort_order },
                    { headers: { Authorization: `Bearer ${token}` } }
                );
                toast({ variant: 'success', title: '更新成功' });
            } else {
                await axios.post(`${API_BASE_URL}/regex/`,
                    { ...formData, sort_order: rules.length },
                    { headers: { Authorization: `Bearer ${token}` } }
                );
                toast({ variant: 'success', title: '创建成功' });
            }
            setIsDialogOpen(false);
            fetchRules();
            resetForm();
        } catch (error) {
            toast({ variant: 'error', title: editingRule ? '更新失败' : '创建失败' });
        }
    };

    const handleDelete = async (id: number) => {
        if (!await confirm({ title: "删除规则", description: "确定要删除此正则规则吗？", confirmText: "删除" })) return;
        const token = localStorage.getItem('token');
        try {
            await axios.delete(`${API_BASE_URL}/regex/${id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            fetchRules();
            toast({ variant: 'success', title: '删除成功' });
        } catch (error) {
            toast({ variant: 'error', title: '删除失败' });
        }
    };

    const resetForm = () => {
        setEditingRule(null);
        setFormData({ name: '', pattern: '', replacement: '', type: 'pre', is_active: true });
    };

    const openEdit = (rule: RegexRule) => {
        setEditingRule(rule);
        setFormData({
            name: rule.name,
            pattern: rule.pattern,
            replacement: rule.replacement,
            type: rule.type,
            is_active: rule.is_active
        });
        setIsDialogOpen(true);
    };

    const handleExport = () => {
        // 导出所有正则，格式化JSON
        const exportData = rules.map(rule => ({
            name: rule.name,
            type: 'regex',
            creator_username: (rule as any).creator_username || 'unknown', // creator_username is not in the interface, keep as any for now
            created_at: format(toZonedTime(new Date(rule.created_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
            updated_at: format(toZonedTime(new Date(rule.updated_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
            enabled: rule.is_active,
            content: {
                type: rule.type,
                pattern: rule.pattern,
                replacement: rule.replacement,
            }
        }));

        exportToJSON(exportData, 'any-api-regex-rules');
        toast({ variant: 'success', title: '导出成功' });
    };

    // 单个正则导出
    const handleExportSingle = (rule: RegexRule) => {
        const exportData = {
            name: rule.name,
            type: 'regex',
            creator_username: (rule as any).creator_username || 'unknown', // creator_username is not in the interface, keep as any for now
            created_at: format(toZonedTime(new Date(rule.created_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
            updated_at: format(toZonedTime(new Date(rule.updated_at), 'Asia/Shanghai'), 'yyyy-MM-dd HH:mm:ss'),
            enabled: rule.is_active,
            content: {
                type: rule.type,
                pattern: rule.pattern,
                replacement: rule.replacement,
            }
        };

        exportToJSON(exportData, `any-api-regex-${rule.name}`);
        toast({ variant: 'success', title: '导出成功' });
    };

    const handleImport = async () => {
        try {
            const importedData = await importFromJSON<any>();
            const token = localStorage.getItem('token');
            let rulesToImport = [];

            // Case 1: Import from a preset file
            if (importedData.type === 'preset' && importedData.content && Array.isArray(importedData.content.regex)) {
                rulesToImport = importedData.content.regex;
            }
            // Case 2: Import from a regex file (array or single object)
            else {
                rulesToImport = Array.isArray(importedData) ? importedData : [importedData];
            }

            if (rulesToImport.length === 0) {
                toast({
                    variant: 'info',
                    title: '没有可导入的规则',
                    description: '文件中未找到兼容的正则规则。',
                });
                return;
            }

            for (const rule of rulesToImport) {
                // Preset files have a different structure
                const ruleContent = rule.content || {};
                await axios.post(`${API_BASE_URL}/regex/`, {
                    name: rule.name,
                    pattern: ruleContent.pattern,
                    replacement: ruleContent.replacement,
                    type: ruleContent.type,
                    is_active: rule.enabled,
                    sort_order: rules.length + rulesToImport.indexOf(rule),
                }, {
                    headers: { Authorization: `Bearer ${token}` }
                });
            }

            fetchRules();
            toast({
                variant: 'success',
                title: '导入成功',
                description: `成功导入 ${rulesToImport.length} 条规则`,
            });
        } catch (error) {
            toast({ variant: 'error', title: '导入失败', description: '文件格式可能不兼容' });
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-end items-center">
                <div className="flex gap-2">
                    <Button onClick={handleExport} variant="outline" size="sm">
                        <Download className="w-4 h-4 mr-2" />
                        导出
                    </Button>
                    <Button onClick={handleImport} variant="outline" size="sm">
                        <Upload className="w-4 h-4 mr-2" />
                        导入
                    </Button>
                    <Dialog open={isDialogOpen} onOpenChange={(open) => {
                        setIsDialogOpen(open);
                        if (!open) resetForm();
                    }}>
                        <DialogTrigger asChild>
                            <Button size="sm">
                                <Plus className="w-4 h-4 mr-2" />
                                添加规则
                            </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                            <DialogHeader>
                                <DialogTitle>{editingRule ? '编辑规则' : '新建规则'}</DialogTitle>
                                <DialogDescription>
                                    支持标准正则表达式语法和捕获组替换（$1, $2, ...）
                                </DialogDescription>
                            </DialogHeader>
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div className="space-y-2">
                                    <Label htmlFor="name">名称</Label>
                                    <Input
                                        id="name"
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        placeholder="例如：过滤敏感词"
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="type">类型</Label>
                                    <select
                                        id="type"
                                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                                        value={formData.type}
                                        onChange={(e) => setFormData({ ...formData, type: e.target.value as 'pre' | 'post' })}
                                    >
                                        <option value="pre">预处理 (处理用户请求)</option>
                                        <option value="post">后处理 (处理AI响应)</option>
                                    </select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="pattern">正则表达式模式</Label>
                                    <Input
                                        id="pattern"
                                        value={formData.pattern}
                                        onChange={(e) => setFormData({ ...formData, pattern: e.target.value })}
                                        placeholder="例如：\b(你好|hello)\b"
                                        className="font-mono"
                                        required
                                    />
                                    <p className="text-xs text-muted-foreground">
                                        💡 支持捕获组：使用 <code className="bg-muted px-1 rounded">()</code> 捕获内容
                                    </p>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="replacement">替换内容</Label>
                                    <Textarea
                                        id="replacement"
                                        value={formData.replacement}
                                        onChange={(e) => setFormData({ ...formData, replacement: e.target.value })}
                                        placeholder="例如：$1 世界  或  ***"
                                        className="font-mono h-20"
                                    />
                                    <div className="text-xs text-muted-foreground space-y-1">
                                        <p>💡 <strong>捕获组引用：</strong></p>
                                        <ul className="list-disc list-inside pl-2">
                                            <li><code className="bg-muted px-1 rounded">$1</code> - 第一个捕获组</li>
                                            <li><code className="bg-muted px-1 rounded">$2</code> - 第二个捕获组</li>
                                            <li>示例：模式 <code className="bg-muted px-1 rounded">(hello) (world)</code> → 替换 <code className="bg-muted px-1 rounded">$2 $1</code> → 输出 "world hello"</li>
                                        </ul>
                                    </div>
                                </div>
                                <DialogFooter className="sm:justify-between">
                                    <div className="flex items-center gap-2">
                                        <Switch
                                            id="active"
                                            checked={formData.is_active}
                                            onCheckedChange={(checked) => setFormData({ ...formData, is_active: checked })}
                                        />
                                        <Label htmlFor="active">启用此规则</Label>
                                    </div>
                                    <Button type="submit">保存</Button>
                                </DialogFooter>
                            </form>
                        </DialogContent>
                    </Dialog>
                </div>
            </div>

            <div className="bg-card border rounded-lg overflow-hidden">
                {rules.length === 0 ? (
                    <div className="p-8 text-center text-muted-foreground">
                        <p className="text-sm">暂无规则</p>
                        <p className="text-xs mt-1">点击"添加规则"创建第一条规则</p>
                    </div>
                ) : (
                    <DndContext
                        sensors={sensors}
                        collisionDetection={closestCenter}
                        onDragEnd={handleDragEnd}
                    >
                        <SortableContext
                            items={rules.map((r) => r.id)}
                            strategy={verticalListSortingStrategy}
                        >
                            <div>
                                {rules.map((rule) => (
                                    <SortableRuleItem
                                        key={rule.id}
                                        rule={rule}
                                        onEdit={openEdit}
                                        onDelete={handleDelete}
                                        onExportSingle={handleExportSingle}
                                        onToggle={handleToggleRule}
                                    />
                                ))}
                            </div>
                        </SortableContext>
                    </DndContext>
                )}
            </div>
        </div>
    );
}
