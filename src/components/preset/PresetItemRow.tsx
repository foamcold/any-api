import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { GripVertical, Pencil, Trash2, Copy, Cpu, User, Bot } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { PresetItem } from '@/services/presetService';
import { Switch } from '@/components/ui/switch';
import { confirm } from '@/components/ui/ConfirmDialog';

interface PresetItemRowProps {
    item: PresetItem;
    onEdit: (item: PresetItem) => void;
    onDelete: (itemId: number) => void;
    onDuplicate: (item: PresetItem) => void;
    onToggle: (item: PresetItem, enabled: boolean) => void;
}

const roleIcons = {
    system: <Cpu className="w-5 h-5 text-muted-foreground" />,
    user: <User className="w-5 h-5 text-muted-foreground" />,
    assistant: <Bot className="w-5 h-5 text-muted-foreground" />,
};

const typeLabels = {
    normal: '普通消息',
    history: '历史消息',
    user_input: '用户输入',
};

const typeColors = {
    normal: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-100',
    history: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-100',
    user_input: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-100',
};

export function PresetItemRow({ item, onEdit, onDelete, onDuplicate, onToggle }: PresetItemRowProps) {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging,
    } = useSortable({ id: item.id });

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
                "flex items-center gap-3 p-4 rounded-md border bg-card hover:shadow-md transition-colors group",
                isDragging && "shadow-lg ring-2 ring-primary"
            )}
        >
            {/* 拖动手柄 */}
            <div
                {...attributes}
                {...listeners}
                className="cursor-grab active:cursor-grabbing opacity-50 hover:opacity-100"
            >
                <GripVertical className="w-5 h-5" />
            </div>

            {/* 角色图标 */}
            <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
                {roleIcons[item.role]}
            </div>

            {/* 内容区域 */}
            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">{item.name}</span>
                    <span className={cn("px-2 py-0.5 rounded-full text-xs", typeColors[item.type])}>
                        {typeLabels[item.type]}
                    </span>
                    {item.enabled === false && (
                        <span className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded">未启用</span>
                    )}
                </div>
                <p className="text-sm text-muted-foreground truncate">
                    {item.content || '(无内容)'}
                </p>
            </div>

            {/* 开关和操作按钮 */}
            <div className="flex items-center gap-2">
                <Switch
                    checked={item.enabled !== false}
                    onCheckedChange={(checked) => onToggle(item, checked)}
                    onClick={(e) => e.stopPropagation()}
                />

                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => onEdit(item)}
                    >
                        <Pencil className="w-4 h-4" />
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => onDuplicate(item)}
                    >
                        <Copy className="w-4 h-4" />
                    </Button>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive"
                        onClick={async () => {
                            if (item.id && await confirm({ title: "删除条目", description: "确定要删除这个条目吗？", confirmText: "删除" })) {
                                onDelete(item.id);
                            }
                        }}
                    >
                        <Trash2 className="w-4 h-4" />
                    </Button>
                </div>
            </div>
        </div>
    );
}
