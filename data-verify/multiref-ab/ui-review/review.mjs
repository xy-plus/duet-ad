import fs from 'node:fs';
import {spawnSync} from 'node:child_process';
import {chromium} from '/home/xy/.npm/_npx/420ff84f11983ee5/node_modules/playwright/index.mjs';

const secretText = fs.readFileSync('/home/xy/.config/duet-ad1/service.env', 'utf8');
const tokenLine = secretText.match(/^(?:export\s+)?ACCESS_TOKEN=(.*)$/m);
if (!tokenLine) throw new Error('protected access token not found');
let token = tokenLine[1].trim();
if ((token.startsWith('"') && token.endsWith('"')) || (token.startsWith("'") && token.endsWith("'"))) token = token.slice(1,-1);
function ffmpeg(args) {
  const result = spawnSync('/usr/bin/ffmpeg', ['-threads','1', ...args], {maxBuffer: 4*1024*1024});
  if (result.status !== 0) throw new Error('local fixture encoder failed');
  return result.stdout;
}
const images = ['red','blue','green','yellow','white'].map((color,index) => ({
  name: `view-${index+1}.png`, mimeType:'image/png',
  buffer: ffmpeg(['-f','lavfi','-i',`color=c=${color}:s=64x64`,'-frames:v','1','-f','image2pipe','-vcodec','png','pipe:1']),
}));
const video = {name:'review-source.mp4',mimeType:'video/mp4',buffer:ffmpeg([
  '-f','lavfi','-i','color=c=gray:s=64x64:d=1:r=10','-pix_fmt','yuv420p',
  '-movflags','frag_keyframe+empty_moov','-f','mp4','pipe:1',
])};
const reports=[];
const browser=await chromium.launch({headless:true,executablePath:'/home/xy/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome',args:['--no-sandbox','--disable-dev-shm-usage','--renderer-process-limit=1']});
try {
  for (const target of [
    {name:'A',url:'https://127.0.0.1:3261',screenshot:'/home/xy/duet-ad1-h3-max-r2v-multiref-common/data-verify/multiref-ab/ui-review/A-four-images.png'},
    {name:'B',url:'https://127.0.0.1:3271',screenshot:'/home/xy/duet-ad1-h3-max-r2v-multiref-common/data-verify/multiref-ab/ui-review/B-four-images.png'},
  ]) {
    const context=await browser.newContext({ignoreHTTPSErrors:true,viewport:{width:1440,height:1080}});
    await context.addInitScript(value=>localStorage.setItem('cvs_token',value),token);
    const page=await context.newPage();
    const report={instance:target.name,checks:[],posts:[],errors:[]};
    reports.push(report);
    page.on('pageerror',error=>report.errors.push(error.message));
    await page.route('**/*',async route=>{
      const request=route.request();
      if (!['GET','HEAD','OPTIONS'].includes(request.method())) {
        const body=request.postDataBuffer()?.toString('latin1')||'';
        const fields=[...body.matchAll(/Content-Disposition: form-data; name="([^"]+)"(?:; filename="([^"]*)")?/g)].map(match=>({field:match[1],filename:match[2]||null}));
        report.posts.push({path:new URL(request.url()).pathname,method:request.method(),fields});
        await route.fulfill({status:422,contentType:'application/json',body:JSON.stringify({detail:{code:'ui_review_intercepted',message:'Local browser verification: request was intercepted'}})});
        return;
      }
      await route.continue();
    });
    const check=(name,passed,detail={})=>report.checks.push({name,passed,...detail});
    try {
      await page.goto(target.url,{waitUntil:'networkidle',timeout:30000});
      await page.locator('#replacement-select').waitFor({state:'visible',timeout:15000});
      await page.locator('input[name="source-mode"][value="upload"]').check();
      await page.locator('#file-input').setInputFiles(video);
      await page.locator('#replacement-instruction').fill('Use the same product from these ordered reference views');
      for (let count=1;count<=4;count++) {
        await page.locator('#replacement-image-input').setInputFiles(images[count-1]);
        const chips=page.locator('#replacement-image-list .reference-image-chip');
        await page.waitForFunction(expected=>document.querySelectorAll('#replacement-image-list .reference-image-chip').length===expected,count);
        const labels=await page.locator('#replacement-image-list .reference-image-chip > span').allTextContents();
        const loaded=await chips.locator('img').evaluateAll(items=>items.every(img=>img.complete&&img.naturalWidth>0));
        check(`select_${count}_numbered_previews`,labels.length===count&&loaded&&labels.every((label,index)=>label.includes(`参考图 ${index+1}`)),{labels,images_loaded:loaded});
        check(`select_${count}_filenames`,labels.every((label,index)=>label.includes(images[index].name)),{labels});
        const before=report.posts.length;
        await page.locator('#send-btn').click();
        await page.waitForFunction(()=>!document.querySelector('#send-btn').disabled,{timeout:10000});
        const post=report.posts[before];
        check(`submit_${count}_ordered_multipart_intercepted`,!!post&&post.path==='/api/conversations'&&JSON.stringify(post.fields.filter(field=>field.field==='replacement_image').map(field=>field.filename))===JSON.stringify(images.slice(0,count).map(image=>image.name)),{captured:!!post});
      }
      await page.screenshot({path:target.screenshot,fullPage:true});
      await page.locator('#replacement-image-input').setInputFiles(images[4]);
      const oversizedError=await page.locator('#composer-error').textContent();
      check('fifth_image_preserves_existing_four',await page.locator('#replacement-image-list .reference-image-chip').count()===4&&oversizedError.includes('4'),{message:oversizedError});
      await page.getByRole('button',{name:'移除参考图 2',exact:true}).click();
      let labels=await page.locator('#replacement-image-list .reference-image-chip > span').allTextContents();
      check('delete_second_renumbers',labels.length===3&&labels.every((label,index)=>label.includes(`参考图 ${index+1}`)),{labels});
      await page.locator('#replacement-image-input').setInputFiles(images[1]);
      labels=await page.locator('#replacement-image-list .reference-image-chip > span').allTextContents();
      const before=report.posts.length;
      await page.locator('#send-btn').click();
      await page.waitForFunction(()=>!document.querySelector('#send-btn').disabled,{timeout:10000});
      const names=report.posts[before]?.fields.filter(field=>field.field==='replacement_image').map(field=>field.filename);
      check('reselect_appends_and_submits_new_order',JSON.stringify(names)===JSON.stringify(['view-1.png','view-3.png','view-4.png','view-2.png']),{filenames:names});
      await page.locator('#new-chat-btn').click();
      check('new_project_resets_references',await page.locator('#replacement-image-list .reference-image-chip').count()===0&&(await page.locator('#replacement-instruction').inputValue())==='');
      await page.locator('#replacement-image-input').setInputFiles(images);
      check('initial_five_images_rejected_atomically',await page.locator('#replacement-image-list .reference-image-chip').count()===0);
      check('all_mutating_requests_intercepted',report.posts.length===5&&report.posts.every(post=>post.path==='/api/conversations'));
    } catch(error) {
      report.errors.push(String(error.message).split('\n').slice(0,3).join(' '));
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
}
fs.writeFileSync('/home/xy/duet-ad1-h3-max-r2v-multiref-common/data-verify/multiref-ab/ui-review/report.json',JSON.stringify(reports,null,2));
console.log(JSON.stringify(reports));
