# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\clyde\\Special Problem\\AI Cephalometric Model_Xenodent\\web\\dashboard.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\clyde\\Special Problem\\AI Cephalometric Model_Xenodent\\checkpoints', 'checkpoints'), ('C:\\Users\\clyde\\Special Problem\\AI Cephalometric Model_Xenodent\\dataset', 'dataset'), ('C:\\Users\\clyde\\Special Problem\\AI Cephalometric Model_Xenodent\\src', 'src')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CephalometricAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CephalometricAI',
)
