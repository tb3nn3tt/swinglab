import sys, os, shutil, json
standalone = '--standalone' in sys.argv
s=open('src/page.html').read()
core=open('src/core.js').read().replace("if (typeof module !== 'undefined') module.exports","if (typeof module !== 'undefined' && module.exports) module.exports")
ui=open('src/ui.js').read().replace('/*COMPARE*/',open('src/compare.js').read()).replace('/*CAMERA*/', open('src/camera.js').read() if standalone else '')
s=s.replace('/*CORE*/',core).replace('/*UI*/',ui)
if not standalone:
    open('index.html','w').write(s); sys.exit()
reset = """<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>"""
head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>SwingLab</title>
<meta name="description" content="Film a swing or pitch, track every joint and the bat, and see what to fix.">
<meta name="theme-color" content="#0B121B">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="SwingLab">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" href="icons/icon-192.png">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
{reset}
</head>
<body>
"""
s = s.replace('<title>SwingLab</title>\n','',1)
d='dist'; shutil.rmtree(d, ignore_errors=True); os.makedirs(d+'/models'); os.makedirs(d+'/icons')
open(d+'/index.html','w').write(head + s + '\n</body>\n</html>\n')
for f in ['model.json','movenet-lightning-weights.wasm']: shutil.copy('models/'+f, d+'/models/'+f)
for f in os.listdir('icons'): shutil.copy('icons/'+f, d+'/icons/'+f)
json.dump({"name":"SwingLab","short_name":"SwingLab","description":"Swing and pitching motion coach","start_url":"./","scope":"./","display":"standalone","orientation":"portrait","background_color":"#0B121B","theme_color":"#0B121B",
  "icons":[{"src":"icons/icon-192.png","sizes":"192x192","type":"image/png"},{"src":"icons/icon-512.png","sizes":"512x512","type":"image/png"},{"src":"icons/icon-512-maskable.png","sizes":"512x512","type":"image/png","purpose":"maskable"}]}, open(d+'/manifest.webmanifest','w'), indent=1)
shutil.copy('src/sw.js', d+'/sw.js')
open(d+'/.nojekyll','w').write('')
print('built dist')
