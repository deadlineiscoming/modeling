# 从零开始玩转 AI 辅助编程（Session 1：环境部署 + CloudCode 注册订阅）

> 本文档面向国内开发者，从 Python 环境配置到 CloudCode/Claude AI 编程平台注册、充值、终端使用全流程，包含科学上网方案和操作示意图。

---

## 目录
1. [准备工作](#准备工作)
2. [安装 Python 与依赖](#安装-python-与依赖)
3. [科学上网访问 CloudCode / Claude](#科学上网访问-cloudcode--claude)
4. [注册 CloudCode / Claude 账号](#注册-cloudcode--claude-账号)
5. [充值 CloudCode / Claude](#充值-cloudcode--claude)
6. [CloudCode CLI 安装与配置](#cloudcode-cli-安装与配置)
7. [终端使用 CloudCode 示例](#终端使用-cloudcode-示例)
8. [AI 编程实践小技巧](#ai-编程实践小技巧)
9. [常见问题与解决方法](#常见问题与解决方法)

---

## 准备工作
- 已安装 Windows / Mac / Linux 系统
- 科学上网工具（Clash）  
- 美区苹果账号 + 礼品卡（其中一种充值方式）
- HeroSMS 或类似短信接收服务  


---

## 安装 Python 与依赖
1. 下载 Python 安装包 [Python 官网](https://www.python.org/)  
2. 安装时勾选 **Add Python to PATH**
3. 安装依赖：
```bash
pip install --upgrade pip
```
4. 测试 Python：
```bash
python -c "print('Hello AI!')"
```

---

## 科学上网访问 CloudCode / Claude
1. 安装 Clash  
   - [Clash for Windows](https://github.com/clash-verge-rev/clash-verge-rev/releases)  
   - [ClashX for Mac](https://github.com/clashclient/ClashX/releases)
2. 导入配置文件（服务商提供，通常有免费试用，订阅费用可能是20-60元/月不等）  
3. 启用代理并访问 CloudCode / Claude
> ⚠️ 风控提示：
> - 尽量使用与注册手机号一致国家的代理
> - 关闭自动切换节点功能

---

## 注册 CloudCode / Claude 账号
### Claude
1. 打开 Claude 官网 [Anthropic](https://www.anthropic.com/)  
2. 使用邮箱注册（可用 Gmail，建议Gmail绑定自己的手机号，方便验证码验证）
3. 使用海外手机号或者接码平台（我用的是HeroSMS） 获取验证码
> ⚠️ 注意：
> 接码平台提供的一次性号码无法长期使用。
> 如果触发二次验证，可能无法再次接收验证码。

### CloudCode
1. 打开官网 → Sign Up  
2. 验证邮箱（上一步注册用的邮箱） → 登录

---

## 充值 CloudCode / Claude
免费token使用量较少，一般需要开pro会员才勉强够用，价格大约是20美元/月
充值方式有一定门槛，简单来说cloude基本倾向于只让美国本土用户使用。
这里我采用苹果美区账号+美区礼品卡充值的方式完成充值，步骤供参考：
1. 使用美区苹果账号登录 App Store下载claude app，此时可以看到plan付费入口，但是点击订阅提示需要绑定信用卡或者paypal。
实测使用招商银行visa信用卡是不能绑定到美区苹果账号的，
paypal也是一样的情况，中国信用卡绑定不了美区paypal（注意中国paypal是不能给美区苹果账号付款的）

2. 使用支付宝的惠出镜来完成美区苹果app store礼品卡充值
打开支付宝，把定位修改为美国旧金山，搜索“惠出境”，打开后进入“大牌礼卡”->pockyt->App Store & iTurns
填入金额进行付款，这里没有手续费，并且按照实时汇率支付人民币。
3.收到兑换码，到appstore中进行兑换。
4. 使用余额订阅会员
再进入app，plan订阅可以使用余额支付了。

---

## CloudCode CLI 安装与配置
https://code.claude.com/docs/zh-CN/overview

---
![img_1.png](img_1.png)

> 🎉 恭喜！你已经完成 Python 环境配置、科学上网访问、CloudCode / Claude 账号注册与会员订阅，并可以在终端使用 AI 辅助编程工具生成代码了！

