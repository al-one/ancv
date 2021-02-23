package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"net/rpc"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spiral/goridge/v2"
	"github.com/urfave/cli/v2"
	"gocv.io/x/gocv"
	"gocv.io/x/gocv/contrib"
)

type App struct{}

// 物体识别
type DetectReq struct {
	Input   string `json:"input"`
	Output  string `json:"output"`
	CocoDir string `json:"coco-dir"`
}

type DetectRsp struct {
	List    []DetectObj `json:"list"`
	Runtime string
}

type DetectObj struct {
	ClassId    int     `json:"cid"`
	ClassName  string  `json:"class"`
	Confidence float32 `json:"confidence"`
	Left       int     `json:"left"`
	Top        int     `json:"top"`
	Right      int     `json:"right"`
	Bottom     int     `json:"bottom"`
}

func (a *App) Detect(req DetectReq, rsp *DetectRsp) error {
	startTime := time.Now()
	img := gocv.IMRead(req.Input, gocv.IMReadColor)
	defer img.Close()
	if img.Empty() {
		log.Printf("Invalid input image: %s", req.Input)
		return nil
	}

	cocoDir := req.CocoDir
	if cocoDir == "" {
		baseDir := "./data"
		cocoDir = baseDir + "/ssdlite_mobilenet_v2_coco"
	}
	modFile := cocoDir + "/frozen_inference_graph.pb"
	cfgFile := cocoDir + "/ssdlite_mobilenet_v2_coco.pbtxt"

	classes := make([]string, 0)
	file, err := os.Open(cocoDir + "/classes.txt")
	defer file.Close()
	if err != nil {
		log.Fatal(err)
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		classes = append(classes, strings.TrimSpace(scanner.Text()))
	}

	cvNet := gocv.ReadNet(modFile, cfgFile)
	defer cvNet.Close()
	if cvNet.Empty() {
		log.Printf("Error reading network model from : %v %v", modFile, cfgFile)
	}

	var ratio float64
	var mean gocv.Scalar
	var swapRGB bool
	if filepath.Ext(modFile) == ".caffemodel" {
		ratio = 1.0
		mean = gocv.NewScalar(104, 177, 123, 0)
		swapRGB = false
	} else {
		ratio = 1.0 / 127.5
		mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)
		swapRGB = true
	}

	blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, swapRGB, false)
	cvNet.SetInput(blob, "")
	mat := cvNet.Forward("")
	idx := 0
	for i := 0; i < mat.Total(); i += 7 {
		classId := int(mat.GetFloatAt(0, i+1))
		confidence := mat.GetFloatAt(0, i+2)
		if confidence > 0.3 {
			idx++
			l := int(mat.GetFloatAt(0, i+3) * float32(img.Cols()))
			t := int(mat.GetFloatAt(0, i+4) * float32(img.Rows()))
			r := int(mat.GetFloatAt(0, i+5) * float32(img.Cols()))
			b := int(mat.GetFloatAt(0, i+6) * float32(img.Rows()))
			obj := DetectObj{
				ClassId:    classId,
				ClassName:  classes[classId],
				Confidence: confidence,
				Left:       l,
				Top:        t,
				Right:      r,
				Bottom:     b,
			}
			gocv.Rectangle(&img, image.Rect(l, t, r, b), color.RGBA{G: 255}, 2)
			gocv.PutText(
				&img,
				fmt.Sprintf("%d %s(%d)", idx, obj.ClassName, classId),
				image.Point{X: l, Y: t - 10},
				gocv.FontHersheyPlain, 1.2,
				color.RGBA{}, 2,
			)
			rsp.List = append(rsp.List, obj)
		}
	}
	len := len(rsp.List)
	if len > 0 {
		log.Printf("Found %2d objects in %s", len, req.Input)
	}
	if req.Output != "" {
		gocv.IMWrite(req.Output, img)
	}
	dur := float32(time.Since(startTime)) / float32(time.Second)
	rsp.Runtime = fmt.Sprintf("%.6f", dur)
	return nil
}

// 人脸识别
type RecognizeReq struct {
	Input       string `json:"input"`
	Output      string `json:"output"`
	Storage     string `json:"storage"`
	Recognized  string `json:"recognized"`
	LbpCascades string `json:"lbpcascades"`
	FacesDir    string `json:"faces_dir"`
	Ask         bool   `json:"ask"` // only cli
}

type RecognizeRsp struct {
	List     []RecognizeObj `json:"list"`
	Unknowns []RecognizeObj `json:"unknowns"`
	Runtime  string
}

type RecognizeObj struct {
	contrib.PredictResponse
	Ratio  float32 `json:"ratio"`
	Left   int     `json:"left"`
	Top    int     `json:"top"`
	Right  int     `json:"right"`
	Bottom int     `json:"bottom"`
}

func (a *App) Recognize(req RecognizeReq, rsp *RecognizeRsp) error {
	startTime := time.Now()
	img := gocv.IMRead(req.Input, gocv.IMReadColor)
	defer img.Close()
	if img.Empty() {
		log.Printf("Invalid input image: %s", req.Input)
		return nil
	}

	baseDir := "./data"
	lbpDir := req.LbpCascades
	if lbpDir == "" {
		lbpDir = baseDir + "/lbpcascades"
	}
	datDir := req.Storage
	if datDir == "" {
		datDir = baseDir + "/lbph"
	}

	faceClassifier := gocv.NewCascadeClassifier()
	defer faceClassifier.Close()
	faceClassifier.Load(lbpDir + "/lbpcascade_frontalface.xml")

	faceRecognizer := contrib.NewLBPHFaceRecognizer()
	recognizedFile := req.Recognized
	if recognizedFile == "" {
		recognizedFile = datDir + "/recognized_faces.xml"
	}
	isFirst := true
	_, err := os.Stat(recognizedFile)
	if err == nil || os.IsExist(err) {
		faceRecognizer.LoadFile(recognizedFile)
		isFirst = false
	}

	if req.FacesDir == "" {
		req.FacesDir = "."
	}

	trainCount := 0
	gray := gocv.NewMat()
	defer gray.Close()
	rects := faceClassifier.DetectMultiScale(img)
	len := len(rects)
	if len > 0 {
		log.Printf("Found %2d faces in %s", len, req.Input)
	}
	for i, rect := range rects {
		faceImage := img.Region(rect)
		gocv.CvtColor(faceImage, &gray, gocv.ColorBGRToGray)
		if req.Output != "" {
			gocv.IMWrite(fmt.Sprintf("%s-%d.jpg", req.Output, i), faceImage)
		}

		obj := RecognizeObj{
			Left:   rect.Min.X,
			Top:    rect.Min.Y,
			Right:  rect.Max.X,
			Bottom: rect.Max.Y,
		}
		if !isFirst {
			obj.PredictResponse = faceRecognizer.PredictExtendedResponse(gray)
			maxConfidence := float32(200)
			if obj.Confidence < maxConfidence {
				obj.Ratio = (maxConfidence - obj.Confidence) / maxConfidence * 100
			}
		}
		if obj.Label <= 0 || obj.Ratio < 90 {
			// 开始训练
			if req.Ask {
				msg := "成功识别人脸，但该人脸不能确定身份"
				if obj.Label > 0 {
					msg += fmt.Sprintf("，是%d的可能性有%.3f%%", obj.Label, obj.Ratio)
				}
				fmt.Println(msg)
				tim := time.Now()
				tmpFile := fmt.Sprintf(
					"%s/%s-%d%s",
					req.FacesDir,
					tim.Format("20060102_150405"),
					i+1,
					filepath.Ext(req.Input),
				)
				gocv.IMWrite(tmpFile, faceImage)
				defer func(f string) {
					os.Remove(f)
				}(tmpFile)
				fmt.Println(tmpFile)
				fmt.Print("请标记：")
				fmt.Scanln(&obj.Label)
				if obj.Label > 0 {
					if isFirst {
						faceRecognizer.Train([]gocv.Mat{gray}, []int{int(obj.Label)})
						isFirst = false
					} else {
						faceRecognizer.Update([]gocv.Mat{gray}, []int{int(obj.Label)})
					}
					trainCount++
					log.Printf("成功标记第%d张人脸为%d", i+1, obj.Label)
				} else if obj.Label == -1 {
					log.Println("break train.")
					break
				}
			}
		}
		if obj.Label > 0 {
			rsp.List = append(rsp.List, obj)
		} else {
			rsp.Unknowns = append(rsp.Unknowns, obj)
		}
		gocv.Rectangle(&img, rect, color.RGBA{R: 255}, 2)
	}
	if trainCount > 0 {
		faceRecognizer.SaveFile(recognizedFile)
	}
	if req.Output != "" {
		gocv.IMWrite(req.Output, img)
	}
	dur := float32(time.Since(startTime)) / float32(time.Second)
	rsp.Runtime = fmt.Sprintf("%.6f", dur)
	return nil
}

func main() {
	app := &cli.App{
		Name:  "ancv",
		Usage: "by gocv",
		Commands: []*cli.Command{
			{
				Name:    "rpc",
				Aliases: []string{"rpc-serve"},
				Usage:   "Start a rpc serve",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:  "host",
						Usage: "ip:port",
					},
				},
				Action: func(c *cli.Context) error {
					adr := c.String("host")
					if adr == "" {
						adr = ":8861"
					}
					lsn, err := net.Listen("tcp", adr)
					if err != nil {
						panic(err)
					}

					err = rpc.Register(new(App))
					if err != nil {
						panic(err)
					}

					go func() {
						http.ListenAndServe("127.0.0.1:8869", nil)
					}()

					for {
						conn, err := lsn.Accept()
						if err != nil {
							continue
						}
						go rpc.ServeCodec(goridge.NewCodec(conn))
					}
					return nil
				},
			},
			{
				Name:    "det",
				Aliases: []string{"detect"},
				Usage:   "Detect objects",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:     "input",
						Aliases:  []string{"i"},
						Usage:    "input image",
						Required: true,
					},
					&cli.StringFlag{
						Name:    "output",
						Aliases: []string{"o"},
						Usage:   "output image",
					},
					&cli.StringFlag{
						Name:  "coco-dir",
						Usage: "SsdLite coco dir",
					},
				},
				Action: func(c *cli.Context) error {
					app := App{}
					ret := DetectRsp{}
					app.Detect(DetectReq{
						Input:  c.String("input"),
						Output: c.String("output"),
					}, &ret)
					jss, _ := json.MarshalIndent(ret, "", "  ")
					log.Printf("result: %s", string(jss))
					return nil
				},
			},
			{
				Name:    "rec",
				Aliases: []string{"recognize"},
				Usage:   "Recognize Faces",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:     "input",
						Aliases:  []string{"i"},
						Usage:    "input image",
						Required: true,
					},
					&cli.StringFlag{
						Name:    "output",
						Aliases: []string{"o"},
						Usage:   "output image",
					},
					&cli.StringFlag{
						Name:    "storage",
						Aliases: []string{"s"},
						Usage:   "storage dir",
					},
					&cli.StringFlag{
						Name:  "recognized",
						Usage: "recognized file",
					},
					&cli.StringFlag{
						Name:  "lbpcascades",
						Usage: "lbpcascades dir",
					},
					&cli.StringFlag{
						Name:  "faces-dir",
						Usage: "faces image dir",
					},
					&cli.BoolFlag{
						Name:    "ask",
						Aliases: []string{"a"},
						Usage:   "Ask label for unrecognized",
					},
				},
				Action: func(c *cli.Context) error {
					app := App{}
					ret := RecognizeRsp{}
					app.Recognize(RecognizeReq{
						Input:       c.String("input"),
						Output:      c.String("output"),
						Storage:     c.String("storage"),
						Recognized:  c.String("recognized"),
						LbpCascades: c.String("lbpcascades"),
						FacesDir:    c.String("faces-dir"),
						Ask:         c.Bool("ask"),
					}, &ret)
					jss, _ := json.MarshalIndent(ret, "", "  ")
					log.Printf("result: %s", string(jss))
					return nil
				},
			},
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		log.Fatal(err)
	}
}
