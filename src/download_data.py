"""
自动下载 MIT-BIH Arrhythmia Database 到 data/raw/mitdb/
来源：https://physionet.org/content/mitdb/1.0.0/

使用 wfdb 官方 API；如果已下载会跳过。
"""
import os
import sys
import wfdb

# 让脚本从任意位置运行都能找到 config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, DS1_RECORDS, DS2_RECORDS


def download_mitdb(dl_dir: str = RAW_DIR) -> None:
    """下载完整 MIT-BIH 数据集"""
    os.makedirs(dl_dir, exist_ok=True)
    target = sorted(set(DS1_RECORDS + DS2_RECORDS))

    # 检查是否已经下载
    have = []
    miss = []
    for rec in target:
        hea = os.path.join(dl_dir, f"{rec}.hea")
        dat = os.path.join(dl_dir, f"{rec}.dat")
        atr = os.path.join(dl_dir, f"{rec}.atr")
        if os.path.exists(hea) and os.path.exists(dat) and os.path.exists(atr):
            have.append(rec)
        else:
            miss.append(rec)

    print(f"[download] 已存在 {len(have)} 条 / 缺失 {len(miss)} 条")
    if not miss:
        print("[download] 所有目标记录均已下载，跳过")
        return

    print(f"[download] 开始下载 MIT-BIH 到 {dl_dir} ...")
    print(f"[download] PhysioNet 国内网络不稳定时可能慢，请耐心等待")

    try:
        # 使用官方批量下载 API（推荐）
        wfdb.dl_database(
            'mitdb',
            dl_dir=dl_dir,
            records=[str(r) for r in miss],
            overwrite=False,
        )
        print("[download] 下载完成")
    except Exception as e:
        print(f"[download] 批量下载失败：{e}")
        print("[download] 切换为逐条下载...")
        for rec in miss:
            try:
                wfdb.dl_files(
                    'mitdb',
                    dl_dir=dl_dir,
                    files=[f"{rec}.hea", f"{rec}.dat", f"{rec}.atr"],
                )
                print(f"  - {rec} OK")
            except Exception as ee:
                print(f"  - {rec} 失败：{ee}")

    # 二次验证
    miss2 = []
    for rec in target:
        if not all(os.path.exists(os.path.join(dl_dir, f"{rec}.{ext}"))
                   for ext in ('hea', 'dat', 'atr')):
            miss2.append(rec)
    if miss2:
        print(f"[download] 警告：仍有 {len(miss2)} 条未下载：{miss2}")
        print("[download] 请检查网络或手动到 https://physionet.org/content/mitdb/1.0.0/ 下载")
    else:
        print("[download] 全部下载完成")


if __name__ == "__main__":
    download_mitdb()
