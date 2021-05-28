const express = require('express')

// для запуска питона
const { spawn } = require('child_process')

// для версионирования питона под aws ec2
require('dotenv').config()

// для скачивания картинки
const multer = require('multer');
const sharp = require('sharp');

const PORT = 3000
const app = express()

app.use(express.static('.'))

app.listen(PORT,()=>{
	console.log(`Server listens on port ${PORT}`)
})

app.get('/', (req,res)=>{
    res.sendFile(path.resolce(__dirname, 'static','index.html'))
})

const upload = multer({
    limits: {
      fileSize: 4 * 1024 * 1024,
    }
  })

async function classify(fname) {
  const child = spawn(process.env.PYVERSION, ["inferring_result.py"]);
  let data = "";
  for await (const chunk of child.stdout) {
      data += chunk;
  }
  let error = "";
  for await (const chunk of child.stderr) {
      error += chunk;
  }
  const exitCode = await new Promise( (resolve, reject) => {
      child.on('close', resolve);
  });

  if( exitCode) {
      throw new Error( `subprocess error exit ${exitCode}, ${error}`);
  }
  return data.toString();
}

app.post('/imageupload',upload.single('image'),async function (req, res) {
  
  // сохранение картинки
    await sharp(req.file.buffer)
  .resize(64, 64)
  .toFile('./pic.png');

  // получение отклика и отправка в браузер
  classify('pic.png').then(
    data=> {          
      res.writeHead(200, "OK", {"Content-type":"text/plain"});
      res.write(data)
      res.end();
    },
    err=> {    
      res.writeHead(200, "OK", {"Content-type":"text/plain"});
      res.write('Картинка очень большая, не удаётся классифицировать')
      res.end()
    }
  );
  
  return res;
})
