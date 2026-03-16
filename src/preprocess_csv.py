"""
CSV 预处理脚本：去重 + 过滤无效消息
在运行 test_csv_final.py 之前执行，避免浪费嵌入 token。

用法：
    python src/preprocess_csv.py
    python src/preprocess_csv.py --pattern "lccc*.csv"
    python src/preprocess_csv.py --inplace   # 直接覆盖原文件（默认输出到 csv_clean/）
"""

import argparse
import csv
import os
from pathlib import Path


# 与 csv_loader.py 保持一致的过滤规则
def is_valid_msg(msg: str, type_name: str) -> bool:
    if not msg or not msg.strip():
        return False
    msg = msg.strip()
    if len(msg) <= 2:
        return False
    if msg.startswith('[') or msg.startswith('表情'):
        return False
    if '动画表情' in type_name:
        return False
    if msg == "I've accepted your friend request. Now let's chat!":
        return False
    if '<msg>' in msg:
        return False
    return True


def process_file(src_path: Path, dst_path: Path):
    with open(src_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)

    # 过滤无效消息
    valid_rows = [r for r in rows if is_valid_msg(r.get('msg', ''), r.get('type_name', ''))]
    filtered = total - len(valid_rows)

    # 去重：以 (CreateTime, msg前200字) 为 key
    seen = set()
    deduped = []
    dup_count = 0
    for r in valid_rows:
        key = (r.get('CreateTime', ''), r.get('msg', '')[:200])
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(r)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    kept = len(deduped)
    print(f"  {src_path.name}: {total} 条 → 过滤 {filtered} 条无效 + 去重 {dup_count} 条 = 保留 {kept} 条"
          f"（节省 {total - kept} 条，{(total - kept) / total * 100:.1f}%）")
    return total, kept


def main():
    parser = argparse.ArgumentParser(description='微信聊天记录 CSV 预处理：去重 + 过滤')
    parser.add_argument('--pattern', default='*.csv', help='文件匹配模式，默认 *.csv')
    parser.add_argument('--inplace', action='store_true', help='直接覆盖原文件（默认输出到 csv_clean/）')
    args = parser.parse_args()

    csv_dir = Path('csv')
    if not csv_dir.exists():
        print('❌ csv/ 目录不存在，请在项目根目录运行')
        return

    files = sorted(csv_dir.glob(args.pattern))
    if not files:
        print(f'❌ 未找到匹配 "{args.pattern}" 的文件')
        return

    out_dir = csv_dir if args.inplace else Path('csv_clean')
    mode = '覆盖原文件' if args.inplace else f'输出到 {out_dir}/'
    print(f'处理 {len(files)} 个文件（{mode}）\n')

    total_in, total_out = 0, 0
    for src in files:
        dst = out_dir / src.name
        t, k = process_file(src, dst)
        total_in += t
        total_out += k

    print(f'\n汇总: {total_in} 条 → {total_out} 条，节省 {total_in - total_out} 条'
          f'（{(total_in - total_out) / total_in * 100:.1f}%）')
    if not args.inplace:
        print(f'✅ 清洗后文件已保存至 {out_dir}/')
        print(f'   运行主脚本时选择 {out_dir}/ 目录下的文件，或将其替换到 csv/ 后再运行')


if __name__ == '__main__':
    main()
