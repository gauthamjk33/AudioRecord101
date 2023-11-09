# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['adfdprogram.py'],
    pathex=[],
    binaries=[],
    datas=[('in_the_wild.joblib', '.'), ('RM-White-Transparent-Logo.png', '.')],
    hiddenimports=['plyer', 'plyer.platforms.win.notification', 'soundcard', 'soundfile', 'PySimpleGUI', 'librosa', 'joblib', 'sklearn', 'sklearn.ensemble._forest', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='adfdprogram',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
