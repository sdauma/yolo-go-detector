package engine

import "image/color"

// ChineseLabelMap 英文标签到中文标签的映射（80个类别）
var ChineseLabelMap = map[string]string{
	"person":         "人员",
	"bicycle":        "自行车",
	"car":            "汽车",
	"motorcycle":     "摩托车",
	"airplane":       "飞机",
	"bus":            "巴士",
	"train":          "火车",
	"truck":          "卡车",
	"boat":           "船",
	"traffic light":  "红绿灯",
	"fire hydrant":   "消防栓",
	"stop sign":      "停车标志",
	"parking meter":  "停车计时器",
	"bench":          "长凳",
	"bird":           "鸟",
	"cat":            "猫",
	"dog":            "狗",
	"horse":          "马",
	"sheep":          "羊",
	"cow":            "牛",
	"elephant":       "大象",
	"bear":           "熊",
	"zebra":          "斑马",
	"giraffe":        "长颈鹿",
	"backpack":       "背包",
	"umbrella":       "雨伞",
	"handbag":        "手提包",
	"tie":            "领带",
	"suitcase":       "行李箱",
	"frisbee":        "飞盘",
	"skis":           "滑雪板",
	"snowboard":      "雪板",
	"sports ball":    "运动球",
	"kite":           "风筝",
	"baseball bat":   "棒球棍",
	"baseball glove": "棒球手套",
	"skateboard":     "滑板",
	"surfboard":      "冲浪板",
	"tennis racket":  "网球拍",
	"bottle":         "瓶子",
	"wine glass":     "酒杯",
	"cup":            "杯子",
	"fork":           "叉子",
	"knife":          "刀",
	"spoon":          "勺子",
	"bowl":           "碗",
	"banana":         "香蕉",
	"apple":          "苹果",
	"sandwich":       "三明治",
	"orange":         "橙子",
	"broccoli":       "西兰花",
	"carrot":         "胡萝卜",
	"hot dog":        "热狗",
	"pizza":          "披萨",
	"donut":          "甜甜圈",
	"cake":           "蛋糕",
	"chair":          "椅子",
	"couch":          "沙发",
	"potted plant":   "盆栽",
	"bed":            "床",
	"dining table":   "餐桌",
	"toilet":         "厕所",
	"tv":             "电视",
	"laptop":         "笔记本电脑",
	"mouse":          "鼠标",
	"remote":         "遥控器",
	"keyboard":       "键盘",
	"cell phone":     "手机",
	"microwave":      "微波炉",
	"oven":           "烤箱",
	"toaster":        "烤面包机",
	"sink":           "水槽",
	"refrigerator":   "冰箱",
	"book":           "书",
	"clock":          "钟",
	"vase":           "花瓶",
	"scissors":       "剪刀",
	"teddy bear":     "泰迪熊",
	"hair drier":     "吹风机",
	"toothbrush":     "牙刷",
}

// GetChineseLabel 根据英文标签获取中文标签
func GetChineseLabel(englishLabel string) string {
	if ch, ok := ChineseLabelMap[englishLabel]; ok {
		return ch
	}
	return englishLabel
}

// ClassColors 80个类别的颜色映射（与根目录 main.go 保持一致）
var ClassColors = map[string]color.RGBA{
	"person":         {0, 0, 255, 255},     // 纯红色 - 人物
	"bicycle":        {255, 165, 0, 255},   // 橙色 - 自行车
	"car":            {0, 255, 0, 255},     // 纯绿色 - 汽车
	"motorcycle":     {255, 255, 0, 255},   // 纯黄色 - 摩托车
	"airplane":       {255, 0, 255, 255},   // 洋红色 - 飞机
	"bus":            {0, 255, 255, 255},   // 青色 - 巴士
	"train":          {128, 0, 128, 255},   // 紫色 - 火车
	"truck":          {255, 0, 0, 255},     // 纯蓝色 - 卡车
	"boat":           {0, 128, 255, 255},   // 深天蓝色 - 船
	"traffic light":  {128, 0, 128, 255},   // 紫色 - 红绿灯
	"fire hydrant":   {0, 0, 139, 255},     // 深蓝色 - 消防栓
	"stop sign":      {255, 20, 147, 255},  // 深粉色 - 停车标志
	"parking meter":  {218, 165, 32, 255},  // 金色 - 停车计时器
	"bench":          {139, 69, 19, 255},   // 巧克力色 - 长凳
	"bird":           {238, 130, 238, 255}, // 紫罗兰色 - 鸟
	"cat":            {255, 192, 203, 255}, // 粉色 - 猫
	"dog":            {123, 104, 238, 255}, // 中紫色 - 狗
	"horse":          {255, 69, 0, 255},    // 橙红色 - 马
	"sheep":          {144, 238, 144, 255}, // 浅绿色 - 羊
	"cow":            {240, 230, 140, 255}, // 亚麻色 - 牛
	"elephant":       {128, 128, 0, 255},   // 橄榄色 - 大象
	"bear":           {165, 42, 42, 255},   // 棕色 - 熊
	"zebra":          {255, 255, 255, 255}, // 白色 - 斑马
	"giraffe":        {255, 228, 181, 255}, // 蜜蜂色 - 长颈鹿
	"backpack":       {70, 130, 180, 255},  // 钢蓝色 - 背包
	"umbrella":       {255, 193, 37, 255},  // 金菊色 - 雨伞
	"handbag":        {220, 20, 60, 255},   // 猩红色 - 手提包
	"tie":            {75, 0, 130, 255},    // 深紫色 - 领带
	"suitcase":       {244, 164, 96, 255},  // 沙棕色 - 行李箱
	"frisbee":        {50, 205, 50, 255},   // 石灰绿 - 飞盘
	"skis":           {176, 224, 230, 255}, // 粉蓝色 - 滑雪板
	"snowboard":      {106, 90, 205, 255},  // 紫罗兰色 - 雪板
	"sports ball":    {255, 140, 0, 255},   // 深橙色 - 运动球
	"kite":           {148, 0, 211, 255},   // 深紫色 - 风筝
	"baseball bat":   {165, 42, 42, 255},   // 棕色 - 棒球棍
	"baseball glove": {255, 20, 147, 255},  // 深粉色 - 棒球手套
	"skateboard":     {30, 144, 255, 255},  // 道奇蓝 - 滑板
	"surfboard":      {255, 105, 180, 255}, // 粉红色 - 冲浪板
	"tennis racket":  {0, 255, 127, 255},   // 草绿色 - 网球拍
	"bottle":         {216, 191, 216, 255}, // 薄荷奶油色 - 瓶子
	"wine glass":     {255, 218, 185, 255}, // 桃色 - 酒杯
	"cup":            {255, 182, 193, 255}, // 浅粉色 - 杯子
	"fork":           {112, 128, 144, 255}, // 石板灰 - 叉子
	"knife":          {178, 34, 34, 255},   // 鲜红色 - 刀
	"spoon":          {220, 220, 220, 255}, // 浅灰色 - 勺子
	"bowl":           {255, 222, 173, 255}, // 蜂蜡色 - 碗
	"banana":         {255, 255, 0, 255},   // 纯黄色 - 香蕉
	"apple":          {255, 99, 71, 255},   // 番茄红 - 苹果
	"sandwich":       {184, 134, 11, 255},  // 深卡其色 - 三明治
	"orange":         {255, 165, 0, 255},   // 纯橙色 - 橙子
	"broccoli":       {34, 139, 34, 255},   // 森林绿 - 西兰花
	"carrot":         {255, 140, 0, 255},   // 深橙色 - 胡萝卜
	"hot dog":        {188, 143, 143, 255}, // 石色 - 热狗
	"pizza":          {205, 133, 63, 255},  // 石褐色 - 披萨
	"donut":          {139, 69, 19, 255},   // 巧克力色 - 甜甜圈
	"cake":           {255, 192, 203, 255}, // 粉色 - 蛋糕
	"chair":          {107, 142, 35, 255},  // 黄橄榄绿 - 椅子
	"couch":          {47, 79, 79, 255},    // 暗瓦灰色 - 沙发
	"potted plant":   {34, 139, 34, 255},   // 森林绿 - 盆栽
	"bed":            {255, 105, 180, 255}, // 粉红色 - 床
	"dining table":   {210, 105, 30, 255},  // 巧克力色 - 餐桌
	"toilet":         {175, 238, 238, 255}, // 浅碧绿色 - 厕所
	"tv":             {0, 191, 255, 255},   // 深天蓝色 - 电视
	"laptop":         {95, 158, 160, 255},  // 青铜色 - 笔记本电脑
	"mouse":          {221, 160, 221, 255}, // 蓟色 - 鼠标
	"remote":         {138, 43, 226, 255},  // 蓝紫色 - 遥控器
	"keyboard":       {112, 128, 144, 255}, // 石板灰 - 键盘
	"cell phone":     {219, 112, 147, 255}, // 苍紫罗兰色 - 手机
	"microwave":      {186, 85, 211, 255},  // 紫色 - 微波炉
	"oven":           {139, 0, 0, 255},     // 暗红色 - 烤箱
	"toaster":        {160, 82, 45, 255},   // 木色 - 烤面包机
	"sink":           {0, 139, 139, 255},   // 深青色 - 水槽
	"refrigerator":   {70, 130, 180, 255},  // 钢蓝色 - 冰箱
	"book":           {160, 32, 240, 255},  // 紫色 - 书
	"clock":          {255, 215, 0, 255},   // 金色 - 钟
	"vase":           {216, 191, 216, 255}, // 薄荷奶油色 - 花瓶
	"scissors":       {128, 128, 0, 255},   // 橄榄色 - 剪刀
	"teddy bear":     {210, 105, 30, 255},  // 巧克力色 - 泰迪熊
	"hair drier":     {221, 160, 221, 255}, // 蓟色 - 吹风机
	"toothbrush":     {255, 182, 193, 255}, // 浅粉色 - 牙刷
}

// DefaultBoxColor 默认边界框颜色
var DefaultBoxColor = color.RGBA{128, 128, 128, 255}

// GetClassColor 根据类别名获取对应颜色
func GetClassColor(className string) color.RGBA {
	if c, ok := ClassColors[className]; ok {
		return c
	}
	return DefaultBoxColor
}
