import { useState, useEffect } from 'react';
import axios from 'axios';
import { API_BASE_URL } from '@/utils/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { useToast } from '@/hooks/useToast';
import { Save, Server, Mail, Shield, Users, AlertTriangle } from 'lucide-react';

interface SystemConfig {
    id: number;
    site_name: string;
    server_url: string;
    allow_registration: boolean;
    allow_password_login: boolean;
    require_email_verification: boolean;
    enable_turnstile: boolean;
    enable_captcha: boolean;
    enable_ip_rate_limit: boolean;
    email_whitelist_enabled: boolean;
    email_whitelist: string[];
    smtp_host: string | null;
    smtp_port: number;
    smtp_user: string | null;
    smtp_password: string | null;
    smtp_from: string | null;
    smtp_use_tls: boolean;
    turnstile_site_key: string | null;
    turnstile_secret_key: string | null;
    log_level: string;
}

export default function SystemPage() {
    const [config, setConfig] = useState<SystemConfig | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const { toast } = useToast();

    useEffect(() => {
        fetchConfig();
    }, []);

    const fetchConfig = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await axios.get<SystemConfig>(`${API_BASE_URL}/system/config`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setConfig(response.data);
        } catch (error) {
            toast({
                variant: 'error',
                title: '加载失败',
                description: '无法加载系统配置',
            });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!config) return;

        try {
            setSaving(true);
            const token = localStorage.getItem('token');
            await axios.put(`${API_BASE_URL}/system/config`, config, {
                headers: { Authorization: `Bearer ${token}` }
            });
            toast({
                variant: 'success',
                title: '保存成功',
                description: '系统配置已更新',
            });
        } catch (error) {
            toast({
                variant: 'error',
                title: '保存失败',
                description: '无法更新系统配置',
            });
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return <div className="flex items-center justify-center h-full">加载中...</div>;
    }

    if (!config) {
        return <div className="flex items-center justify-center h-full">无法加载配置</div>;
    }

    return (
        <div className="space-y-6 max-w-4xl">
            <div className="flex justify-end items-center">
                <Button onClick={handleSave} disabled={saving}>
                    <Save className="w-4 h-4 mr-2" />
                    {saving ? '保存中...' : '保存设置'}
                </Button>
            </div>

            {/* 基本设置 */}
            <div className="bg-card border rounded-lg p-6 space-y-4">
                <div className="flex items-center gap-2 mb-4">
                    <Server className="w-5 h-5 text-primary" />
                    <h2 className="text-xl font-semibold">基本设置</h2>
                </div>

                <div className="grid gap-4">
                    <div className="space-y-2">
                        <Label htmlFor="site_name">网站名称</Label>
                        <Input
                            id="site_name"
                            value={config.site_name}
                            onChange={(e) => setConfig({ ...config, site_name: e.target.value })}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="server_url">服务器地址</Label>
                        <Input
                            id="server_url"
                            value={config.server_url}
                            onChange={(e) => setConfig({ ...config, server_url: e.target.value })}
                            placeholder="http://localhost:8000"
                        />
                        <p className="text-xs text-muted-foreground">用于首页和文档展示的服务器地址</p>
                    </div>
                </div>
            </div>

            {/* 日志配置 */}
            <div className="bg-card border rounded-lg p-6 space-y-4">
                <div className="flex items-center gap-2 mb-4">
                    <AlertTriangle className="w-5 h-5 text-primary" />
                    <h2 className="text-xl font-semibold">日志配置</h2>
                </div>

                <div className="grid gap-4">
                    <div className="space-y-2">
                        <Label htmlFor="log_level">日志级别</Label>
                        <select
                            id="log_level"
                            value={config.log_level || 'INFO'}
                            onChange={(e) => setConfig({ ...config, log_level: e.target.value })}
                            className="flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            <option value="INFO">INFO (信息 - 一般信息)</option>
                            <option value="DEBUG">DEBUG (调试 - 详细信息)</option>
                            <option value="WARNING">WARNING (警告 - 潜在问题)</option>
                            <option value="ERROR">ERROR (错误 - 严重问题)</option>
                        </select>
                        <p className="text-xs text-muted-foreground">控制后端输出日志的详细程度，问题排查时请开启 DEBUG</p>
                    </div>
                </div>
            </div>

            {/* 注册配置 */}
            <div className="bg-card border rounded-lg p-6 space-y-4">
                <div className="flex items-center gap-2 mb-4">
                    <Users className="w-5 h-5 text-primary" />
                    <h2 className="text-xl font-semibold">注册配置</h2>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <Label htmlFor="allow_registration">允许新用户注册</Label>
                            <p className="text-sm text-muted-foreground">关闭后，新用户将无法通过任何方式注册</p>
                        </div>
                        <Switch
                            id="allow_registration"
                            checked={config.allow_registration}
                            onCheckedChange={(checked) => setConfig({ ...config, allow_registration: checked })}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <Label htmlFor="allow_password_login">允许密码登录</Label>
                            <p className="text-sm text-muted-foreground">关闭后，仅管理员和超级管理员可使用密码登录</p>
                        </div>
                        <Switch
                            id="allow_password_login"
                            checked={config.allow_password_login}
                            onCheckedChange={(checked) => setConfig({ ...config, allow_password_login: checked })}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <Label htmlFor="require_email_verification">启用邮件验证</Label>
                            <p className="text-sm text-muted-foreground">启用后，新用户注册和重置密码时需要通过邮件验证</p>
                        </div>
                        <Switch
                            id="require_email_verification"
                            checked={config.require_email_verification}
                            onCheckedChange={(checked) => setConfig({ ...config, require_email_verification: checked })}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <Label htmlFor="enable_turnstile">启用Turnstile验证</Label>
                            <p className="text-sm text-muted-foreground">Cloudflare人机验证</p>
                        </div>
                        <Switch
                            id="enable_turnstile"
                            checked={config.enable_turnstile}
                            onCheckedChange={(checked) => setConfig({ ...config, enable_turnstile: checked })}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <Label htmlFor="enable_captcha">启用验证码</Label>
                            <p className="text-sm text-muted-foreground">在发送邮件前需要输入图形验证码</p>
                        </div>
                        <Switch
                            id="enable_captcha"
                            checked={config.enable_captcha}
                            onCheckedChange={(checked) => setConfig({ ...config, enable_captcha: checked })}
                        />
                    </div>

                    <div className="flex items-center justify-between">
                        <div>
                            <Label htmlFor="enable_ip_rate_limit">启用IP注册频率限制</Label>
                            <p className="text-sm text-muted-foreground">防止同一IP地址在短时间内大量发送注册/重置邮件</p>
                        </div>
                        <Switch
                            id="enable_ip_rate_limit"
                            checked={config.enable_ip_rate_limit}
                            onCheckedChange={(checked) => setConfig({ ...config, enable_ip_rate_limit: checked })}
                        />
                    </div>
                </div>
            </div>

            {/* 邮箱白名单和SMTP配置 - 条件渲染 */}
            {config.require_email_verification && (
                <>
                    {/* 邮箱白名单 */}
                    <div className="bg-card border rounded-lg p-6 space-y-4">
                        <div className="flex items-center gap-2 mb-4">
                            <Mail className="w-5 h-5 text-primary" />
                            <h2 className="text-xl font-semibold">邮箱白名单</h2>
                        </div>

                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="email_whitelist_enabled">启用邮箱白名单</Label>
                                    <p className="text-sm text-muted-foreground">只允许特定邮箱后缀注册</p>
                                </div>
                                <Switch
                                    id="email_whitelist_enabled"
                                    checked={config.email_whitelist_enabled}
                                    onCheckedChange={(checked) => setConfig({ ...config, email_whitelist_enabled: checked })}
                                />
                            </div>

                            {config.email_whitelist_enabled && (
                                <div className="space-y-2">
                                    <Label htmlFor="email_whitelist">允许的邮箱后缀（每行一个）</Label>
                                    <Textarea
                                        id="email_whitelist"
                                        value={config.email_whitelist.join('\n')}
                                        onChange={(e) => setConfig({
                                            ...config,
                                            email_whitelist: e.target.value.split('\n').filter(s => s.trim())
                                        })}
                                        className="h-24"
                                        placeholder="qq.com&#10;outlook.com&#10;gmail.com"
                                    />
                                </div>
                            )}

                        </div>
                    </div>

                    {/* SMTP配置 */}
                    <div className="bg-card border rounded-lg p-6 space-y-4">
                        <h2 className="text-xl font-semibold">SMTP 邮件配置</h2>
                        <p className="text-sm text-muted-foreground">
                            配置SMTP服务器以发送邮箱验证邮件
                        </p>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label htmlFor="smtp_host">SMTP 主机</Label>
                                <Input
                                    id="smtp_host"
                                    value={config.smtp_host || ''}
                                    onChange={(e) => setConfig({ ...config, smtp_host: e.target.value })}
                                    placeholder="smtp.gmail.com"
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="smtp_port">SMTP 端口</Label>
                                <Input
                                    id="smtp_port"
                                    type="number"
                                    value={config.smtp_port}
                                    onChange={(e) => setConfig({ ...config, smtp_port: parseInt(e.target.value) || 587 })}
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="smtp_user">SMTP 用户名</Label>
                                <Input
                                    id="smtp_user"
                                    value={config.smtp_user || ''}
                                    onChange={(e) => setConfig({ ...config, smtp_user: e.target.value })}
                                />
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="smtp_password">SMTP 密码</Label>
                                <Input
                                    id="smtp_password"
                                    type="password"
                                    value={config.smtp_password || ''}
                                    onChange={(e) => setConfig({ ...config, smtp_password: e.target.value })}
                                    placeholder="留空则不修改"
                                />
                            </div>

                            <div className="space-y-2 col-span-2">
                                <Label htmlFor="smtp_from">发件人地址</Label>
                                <Input
                                    id="smtp_from"
                                    type="email"
                                    value={config.smtp_from || ''}
                                    onChange={(e) => setConfig({ ...config, smtp_from: e.target.value })}
                                    placeholder="noreply@example.com"
                                />
                            </div>

                            <div className="flex items-center justify-between col-span-2">
                                <Label htmlFor="smtp_use_tls">使用 TLS</Label>
                                <Switch
                                    id="smtp_use_tls"
                                    checked={config.smtp_use_tls}
                                    onCheckedChange={(checked) => setConfig({ ...config, smtp_use_tls: checked })}
                                />
                            </div>
                        </div>
                    </div>
                </>
            )}

            {/* Turnstile配置 */}
            {config.enable_turnstile && (
                <div className="bg-card border rounded-lg p-6 space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                        <Shield className="w-5 h-5 text-primary" />
                        <h2 className="text-xl font-semibold">Cloudflare Turnstile</h2>
                    </div>

                    <div className="grid gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="turnstile_site_key">Site Key</Label>
                            <Input
                                id="turnstile_site_key"
                                value={config.turnstile_site_key || ''}
                                onChange={(e) => setConfig({ ...config, turnstile_site_key: e.target.value })}
                            />
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="turnstile_secret_key">Secret Key</Label>
                            <Input
                                id="turnstile_secret_key"
                                type="password"
                                value={config.turnstile_secret_key || ''}
                                onChange={(e) => setConfig({ ...config, turnstile_secret_key: e.target.value })}
                                placeholder="留空则不修改"
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
