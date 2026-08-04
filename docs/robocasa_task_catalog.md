# RoboCasa365 可用任务中文目录

> 自动整理自 `robocasa/docs/composite_tasks/task_attributes.json`。
>
> 字段说明：`子任务数` 是 RoboCasa catalog 中记录的 `num_subtasks`；`移动` 对应 `moma_required`，表示该任务是否标注为需要移动操作能力。任务名保留 RoboCasa 原始英文类名；`阶段说明` 根据 RoboCasa 原始任务描述整理为中文阶段式说明，用于快速筛选可组合任务。

- 总任务数：365
- 原子任务：65
- 组合任务：300
- 任务类别数：61

## 快速索引

- [原子任务](#原子任务)：65 个
- [饮品加冰](#饮品加冰)：4 个
- [自助餐摆放](#自助餐摆放)：5 个
- [柜子整理](#柜子整理)：7 个
- [调味品整理](#调味品整理)：3 个
- [烘焙](#烘焙)：7 个
- [煮沸](#煮沸)：8 个
- [冲泡饮品](#冲泡饮品)：6 个
- [烤鱼](#烤鱼)：5 个
- [食物切配](#食物切配)：6 个
- [蔬菜切配](#蔬菜切配)：2 个
- [电器清洁](#电器清洁)：3 个
- [水槽清洁](#水槽清洁)：3 个
- [餐桌清理](#餐桌清理)：8 个
- [食物解冻](#食物解冻)：6 个
- [装填上菜容器](#装填上菜容器)：4 个
- [煎制](#煎制)：8 个
- [菜品装饰](#菜品装饰)：5 个
- [装载洗碗机](#装载洗碗机)：2 个
- [冰箱收纳](#冰箱收纳)：8 个
- [制作果汁](#制作果汁)：3 个
- [制作沙拉](#制作沙拉)：2 个
- [制作奶昔/冰沙](#制作奶昔/冰沙)：5 个
- [泡茶](#泡茶)：3 个
- [制作吐司](#制作吐司)：3 个
- [冷冻柜空间管理](#冷冻柜空间管理)：8 个
- [食材计量](#食材计量)：3 个
- [肉类准备](#肉类准备)：2 个
- [微波加热食物](#微波加热食物)：6 个
- [混合食材](#混合食材)：6 个
- [搅拌与混合](#搅拌与混合)：3 个
- [餐具与容器整理](#餐具与容器整理)：3 个
- [回收物整理](#回收物整理)：4 个
- [餐具整理](#餐具整理)：3 个
- [打包午餐](#打包午餐)：4 个
- [装盘](#装盘)：3 个
- [餐食分装](#餐食分装)：7 个
- [准备热巧克力](#准备热巧克力)：2 个
- [准备腌料](#准备腌料)：3 个
- [准备三明治](#准备三明治)：6 个
- [重新加热食物](#重新加热食物)：6 个
- [补充物资](#补充物资)：4 个
- [砧板消毒](#砧板消毒)：4 个
- [台面消毒](#台面消毒)：6 个
- [炒蔬菜](#炒蔬菜)：7 个
- [食物调味](#食物调味)：3 个
- [供应饮品](#供应饮品)：7 个
- [上菜](#上菜)：5 个
- [摆餐桌](#摆餐桌)：13 个
- [炖煮酱汁](#炖煮酱汁)：1 个
- [肉类切配](#肉类切配)：3 个
- [慢炖](#慢炖)：3 个
- [准备零食](#准备零食)：5 个
- [食材分类](#食材分类)：2 个
- [蒸食物](#蒸食物)：3 个
- [蒸蔬菜](#蒸蔬菜)：3 个
- [剩菜收纳](#剩菜收纳)：5 个
- [柜子与抽屉整理](#柜子与抽屉整理)：5 个
- [烤面包](#烤面包)：7 个
- [洗餐具](#洗餐具)：18 个
- [清洗果蔬](#清洗果蔬)：11 个

## 原子任务

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AdjustToasterOvenTemperature` | 1 | 否 | 1. 调高或调低烤面包箱温度。 |
| `AdjustWaterTemperature` | 1 | 否 | 1. 确认水槽水温是热或冷；2. 调节水龙头把手，让水变成相反温度；3. 保持水流开启。 |
| `CheesyBread` | 1 | 否 | 1. 拿起奶酪块；2. 把奶酪放到面包片上。 |
| `CloseBlenderLid` | 1 | 是 | 1. 把盖子稳固放到搅拌机上并关闭。 |
| `CloseCabinet` | 1 | 否 | 1. 关闭柜子。 |
| `CloseDishwasher` | 1 | 否 | 1. 关闭洗碗机。 |
| `CloseDrawer` | 1 | 否 | 1. 关闭抽屉。 |
| `CloseElectricKettleLid` | 1 | 否 | 1. 关闭电水壶盖子。 |
| `CloseFridge` | 1 | 否 | 1. 关闭冰箱。 |
| `CloseFridgeDrawer` | 1 | 否 | 1. 关闭冰箱抽屉。 |
| `CloseMicrowave` | 1 | 是 | 1. 关闭微波炉。 |
| `CloseOven` | 1 | 否 | 1. 关闭烤箱。 |
| `CloseStandMixerHead` | 1 | 是 | 1. 关闭厨师机机头。 |
| `CloseToasterOvenDoor` | 1 | 否 | 1. 关闭烤面包箱门。 |
| `CoffeeServeMug` | 1 | 否 | 1. 从咖啡机出液口下方拿起马克杯；2. 把马克杯放到台面。 |
| `CoffeeSetupMug` | 1 | 否 | 1. 从台面拿起马克杯；2. 把马克杯放到咖啡机出液口下方。 |
| `LowerHeat` | 1 | 是 | 1. 调低水壶加热火力。 |
| `MakeIcedCoffee` | 1 | 否 | 1. 拿起冰块；2. 把冰块放入咖啡杯。 |
| `NavigateKitchen` | 1 | 是 | 1. 导航到指定厨房位置。 |
| `OpenBlenderLid` | 1 | 否 | 1. 取下搅拌机盖子；2. 把盖子放到台面。 |
| `OpenCabinet` | 1 | 否 | 1. 打开柜子。 |
| `OpenDishwasher` | 1 | 否 | 1. 打开洗碗机。 |
| `OpenDrawer` | 1 | 否 | 1. 打开抽屉。 |
| `OpenElectricKettleLid` | 1 | 否 | 1. 打开电水壶盖子。 |
| `OpenFridge` | 1 | 否 | 1. 打开冰箱。 |
| `OpenFridgeDrawer` | 1 | 是 | 1. 打开冰箱抽屉。 |
| `OpenMicrowave` | 1 | 是 | 1. 打开微波炉。 |
| `OpenOven` | 1 | 否 | 1. 打开烤箱。 |
| `OpenStandMixerHead` | 1 | 否 | 1. 打开厨师机机头。 |
| `OpenToasterOvenDoor` | 1 | 否 | 1. 打开烤面包箱门。 |
| `PackDessert` | 1 | 否 | 1. 拿起甜点；2. 把甜点放入装有食物的保鲜盒。 |
| `PickPlaceCabinetToCounter` | 1 | 否 | 1. 从柜子拿起物品；2. 把物品放到台面。 |
| `PickPlaceCounterToBlender` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到搅拌机。 |
| `PickPlaceCounterToCabinet` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到柜子。 |
| `PickPlaceCounterToDrawer` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到抽屉。 |
| `PickPlaceCounterToMicrowave` | 1 | 是 | 1. 从台面拿起物品；2. 把物品放到微波炉。 |
| `PickPlaceCounterToOven` | 1 | 是 | 1. 从台面拿起物品；2. 把物品放到烤箱。 |
| `PickPlaceCounterToSink` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到水槽。 |
| `PickPlaceCounterToStandMixer` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到厨师机。 |
| `PickPlaceCounterToStove` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到炉灶。 |
| `PickPlaceCounterToToasterOven` | 1 | 否 | 1. 从台面拿起物品；2. 把物品放到烤面包箱。 |
| `PickPlaceDrawerToCounter` | 1 | 是 | 1. 从抽屉拿起物品；2. 把物品放到台面。 |
| `PickPlaceFridgeDrawerToShelf` | 1 | 否 | 1. 从冰箱抽屉拿起物品；2. 把物品放到冰箱搁架。 |
| `PickPlaceFridgeShelfToDrawer` | 1 | 否 | 1. 从冰箱搁架拿起物品；2. 把物品放到抽屉。 |
| `PickPlaceMicrowaveToCounter` | 1 | 是 | 1. 从微波炉拿起物品；2. 把物品放到台面。 |
| `PickPlaceSinkToCounter` | 1 | 否 | 1. 从水槽拿起物品；2. 把物品放到台面。 |
| `PickPlaceStoveToCounter` | 1 | 否 | 1. 从炉灶拿起物品；2. 把物品放到台面。 |
| `PickPlaceToasterOvenToCounter` | 1 | 否 | 1. 从烤面包箱拿起物品；2. 把物品放到台面。 |
| `PickPlaceToasterToCounter` | 1 | 否 | 1. 从烤面包机拿起物品；2. 把物品放到台面。 |
| `PreheatOven` | 1 | 否 | 1. 旋转温度旋钮预热烤箱。 |
| `SlideDishwasherRack` | 1 | 否 | 1. 将洗碗机架子推入或拉出到目标位置。 |
| `SlideOvenRack` | 1 | 否 | 1. 将烤箱架子推入或拉出到目标位置。 |
| `SlideToasterOvenRack` | 1 | 否 | 1. 将烤面包箱架子推入或拉出到目标位置。 |
| `StartCoffeeMachine` | 1 | 否 | 1. 按下咖啡机按钮开始出咖啡。 |
| `TurnOffMicrowave` | 1 | 否 | 1. 关闭微波炉。 |
| `TurnOffSinkFaucet` | 1 | 否 | 1. 关闭水龙头。 |
| `TurnOffStove` | 1 | 是 | 1. 关闭炉灶。 |
| `TurnOnBlender` | 1 | 否 | 1. 打开搅拌机。 |
| `TurnOnElectricKettle` | 1 | 否 | 1. 打开电水壶。 |
| `TurnOnMicrowave` | 1 | 否 | 1. 打开微波炉。 |
| `TurnOnSinkFaucet` | 1 | 否 | 1. 打开水龙头。 |
| `TurnOnStove` | 1 | 是 | 1. 打开炉灶。 |
| `TurnOnToaster` | 1 | 否 | 1. 打开烤面包机。 |
| `TurnOnToasterOven` | 1 | 否 | 1. 打开烤面包箱。 |
| `TurnSinkSpout` | 1 | 否 | 1. 把水槽出水口转向左侧或右侧。 |

## 饮品加冰

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `MakeIceLemonade` | 5 | 是 | 1. grab  lemon wedge 从冰箱并 one ice cube 从  ice 碗， 并放置在  glass  lemonade。 |
| `PlaceEqualIceCubes` | 4 | 是 | 1. there  four ice cubes 在  ice 碗；2. 放置 two ice cubes 在 each glass  水。 |
| `PlaceIceInCup` | 2 | 是 | 1. there  two ice cubes 在  ice 碗；2. 放置两个 ice cubes 在  glass 杯子。 |
| `RetrieveIceTray` | 4 | 是 | 1. 打开冷冻柜门， 拿起  ice cube tray 从冷冻柜， 并放置在餐台。 |

## 自助餐摆放

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ArrangeBuffetDessert` | 8 | 是 | 1. 拿起  [*sweets*] 从冰箱并放置在  empty tray 在餐台。 |
| `CutBuffetPizza` | 2 | 是 | 1. 拿起  pizza cutter 从抽屉在  *left/right* 并放置在餐台用于 cutting。 |
| `DivideBuffetTrays` | 16 | 是 | 1. gather  蔬菜从冰箱并放置在  tray 在餐台；2. gather  [*meats*] 从冰箱并放置在  other tray。 |
| `PlaceBeveragesTogether` | 11 | 是 | 1. 放置  drinks 在餐台在  tight cluster。 |
| `TongBuffetSetup` | 4 | 是 | 1. 拿起  tongs 从抽屉在  *left/right* 并放置在餐台 next 到  tray containing  食物。 |

## 柜子整理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BeverageSorting` | 9 | 是 | 1. 分类所有 alcoholic drinks 到 one 柜子， 并 non-alcoholic drinks 到  other。 |
| `GatherTableware` | 4 | 是 | 1. gather 所有 objects 放入 one 柜子并分类  glasses 并 bowls 到 opposite sides。 |
| `ResetCabinetDoors` | 5 | 是 | 1. 关闭所有打开柜子 doors。 |
| `RestockBowls` | 4 | 否 | 1. 打开柜子；2. 从台面拿起碗并放入柜子；3. 关闭柜子。 |
| `RestockPantry` | 2 | 否 | 1. 从台面拿起罐头；2. 按指定位置把罐头放入柜子。 |
| `StackCans` | 2 | 否 | 1. stack two cans 在 top   other two cans 在柜子。 |
| `StockingBreakfastFoods` | 3 | 是 | 1. 拿起  [*packaged foods*] 从台面并放置在  cabinets closest 到。 |

## 调味品整理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `CategorizeCondiments` | 2 | 是 | 1. 放置  shaker 并 condiment bottle 从台面 next 到 their counterparts 在柜子。 |
| `LineUpCondiments` | 7 | 是 | 1. 拿起 two condiments 从柜子并移动在台面 near  炉灶；2. make sure  condiments  lined up 带有 each other， next 到 each other。 |
| `OrganizeCondiments` | 3 | 是 | 1. there  various 物品在台面；2. only 移动  condiments/shakers 放入柜子。 |

## 烘焙

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `CookieDoughPrep` | 7 | 是 | 1. 拿起  egg 并 butter 从冰箱并放置通过碗用于 mixing。 |
| `CoolBakedCake` | 8 | 是 | 1. 拿起  baked cake out   烤箱并放置在盘子用于 cooling；2. 关闭烤箱门。 |
| `CoolBakedCookies` | 5 | 是 | 1. 打开烤箱门并滑动  top 架子 out；2. 拿起  tray 带有 baked cookies out   烤箱并放置在台面用于 cooling。 |
| `CupcakeCleanup` | 2 | 否 | 1. 移动  fresh-baked cupcake off  tray onto  台面， 并放置碗 used 用于 mixing 放入水槽。 |
| `MixCakeFrosting` | 5 | 是 | 1. 加入 butter 并 sugar cubes 到  stand mixer 碗并；2. 打开  stand mixer 到 begin making frosting。 |
| `OrganizeBakingIngredients` | 3 | 是 | 1. gather  eggs 并 milk located 在  vicinity   水槽 area 并放置  right next 到碗。 |
| `PastryDisplay` | 2 | 是 | 1. 放置  pastries 在  plates。 |

## 煮沸

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BoilCorn` | 4 | 是 | 1. 把两段玉米从盘子放入小锅；2. 给小锅盖上盖子开始煮玉米。 |
| `BoilEggs` | 2 | 是 | 1. 从盘子拿起两个鸡蛋；2. 把鸡蛋放入装有沸水的小锅。 |
| `BoilPot` | 4 | 是 | 1. 拿起装水的锅并放到炉灶上；2. 打开炉头开始烧水。 |
| `CoolKettle` | 4 | 是 | 1. 等水壶烧开；2. 关闭炉灶；3. 等待 300 个 timestep；4. 把水壶移动到餐桌。 |
| `FillKettle` | 4 | 是 | 1. 打开柜子；2. 从柜子拿起水壶；3. 把水壶放入水槽。 |
| `HeatMultipleWater` | 4 | 是 | 1. 拿起水壶从柜子并放置在炉头；2. 拿起锅从台面并放置在 another 炉头；3. finally， 打开两个 burners。 |
| `PlaceLidToBoil` | 3 | 是 | 1. 把盖子放到炉灶上的小锅上，加快烧水。 |
| `StartElectricKettle` | 2 | 否 | 1. 关闭电水壶盖子；2. 打开电水壶。 |

## 冲泡饮品

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ArrangeTea` | 3 | 否 | 1. 拿起水壶从台面并放置在  tray；2. 拿起马克杯从柜子并放置在  tray；3. 关闭柜子 doors。 |
| `DeliverBrewedCoffee` | 3 | 是 | 1. 拿起  brewed coffee 从咖啡机并放置  next 到  breakfast 盘子在餐台。 |
| `KettleBoiling` | 2 | 是 | 1. 从台面拿起水壶并放到炉头上；2. 打开炉头开始加热。 |
| `OrganizeCoffeeCondiments` | 2 | 否 | 1. 拿起  *syrup bottle/cinnamon* 并 honey bottle 从柜子并放置  next 到马克杯。 |
| `PrepareCoffee` | 2 | 否 | 1. 拿起马克杯从柜子， 放置  under  咖啡机 dispenser， 并按下启动按钮。 |
| `SweetenCoffee` | 5 | 是 | 1. 拿起  sugar cube 从  saucer 并放置  inside  coffee；2. 然后， grab milk 从冰箱并放置  next 到  coffee 在台面到 sweeten  coffee。 |

## 烤鱼

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `OvenBroilFish` | 3 | 是 | 1. 放置  fish 在  *bottom/top* 架子烤箱， 并滑动  tray back inside；2. 关闭烤箱门。 |
| `PrepareBroilingStation` | 3 | 是 | 1. 拿起  tongs 从台面并放置  near  盘子 next 到烤箱。 |
| `RemoveBroiledFish` | 5 | 是 | 1. 拿起  fish 从烤箱并放置在盘子在台面。 |
| `ToasterOvenBroilFish` | 4 | 是 | 1. 把鱼放到烤面包箱托盘；2. 将托盘推回；3. 关闭烤面包箱门；4. 设置最高温开始烤鱼。 |
| `WashFish` | 4 | 是 | 1. 拿起  fish out   冰箱， briefly 冲洗在水槽， 并放置在  tray next 到水槽。 |

## 食物切配

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ArrangeCuttingFruits` | 7 | 是 | 1. 放置  fruits 在冰箱在砧板通过水槽到 make  fruit 碗；2. leave  non-fruit 物品在冰箱。 |
| `ArrangeVegetables` | 2 | 否 | 1. 拿起  vegetables 从水槽并放置在砧板。 |
| `BreadSetupSlicing` | 1 | 否 | 1. 放置所有 breads 在砧板。 |
| `ClearCuttingBoard` | 3 | 否 | 1. 清空  non-蔬菜 object off  砧板并放置  vegetables onto。 |
| `MeatTransfer` | 3 | 是 | 1. retrieve  [*容器*] 从柜子；2. 放置  raw 肉类放入  [*容器*] 到 avoid contamination。 |
| `OrganizeVegetables` | 2 | 否 | 1. 放置  [*蔬菜1*] 并  [*蔬菜2*] 在 separate cutting boards。 |

## 蔬菜切配

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `CuttingToolSelection` | 1 | 是 | 1. 放置  appropriate cutting tool 用于 cutting  [*fruit/蔬菜*] skin 在砧板。 |
| `GatherCuttingTools` | 3 | 是 | 1. 打开抽屉在  *left/right* 并放置  cutting tools 在砧板。 |

## 电器清洁

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `CleanBlenderJug` | 3 | 是 | 1. 拿起  blender jug 并放置在水槽到准备用于 cleaning。 |
| `PrepFridgeForCleaning` | 4 | 是 | 1. 打开冰箱， 移除水果， 蔬菜， 并肉类， 并放置在任意台面。 |
| `PrepSinkForCleaning` | 7 | 是 | 1. 移动  [*drinkware*] 从 next 到水槽到打开柜子；2. 然后， 移动碗带有食物到冰箱到清空  area 用于 faucet cleaning。 |

## 水槽清洁

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ClearFoodWaste` | 2 | 否 | 1. 移动食物物品 out   水槽并放入  colander。 |
| `ClearSinkArea` | 11 | 是 | 1. 把马克杯和碗移到远离水槽的位置。 |
| `RinseSinkBasin` | 2 | 否 | 1. 打开水槽并 manuever  spout 到清洗所有 locations   水槽 basin。 |

## 餐桌清理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BowlAndCup` | 4 | 是 | 1. 放置杯子 inside  碗在  dining table 并移动碗到任意台面。 |
| `CandleCleanup` | 8 | 是 | 1. 拿起  [*decorations*] 从  dining table 并放置在打开柜子；2. 关闭柜子 after placing  objects。 |
| `ClearReceptaclesForCleaning` | 8 | 是 | 1. 拿起  [*tableware*] 并放置在水槽；2. 打开水。 |
| `ClusterItemsForClearing` | 7 | 是 | 1. 移动  [*tableware1*] 到 cluster 带有  [*tableware2*] 在餐台用于 efficient clearing。 |
| `CondimentCollection` | 2 | 是 | 1. 拿起  [*condiments*] 从台面并放置在柜子。 |
| `DessertAssembly` | 2 | 是 | 1. 放置  [*容器*] 带有 [*dessert*] 并  cupcake onto  tray。 |
| `DrinkwareConsolidation` | 3 | 是 | 1. 拿起  [*drinkware*] 从台面并放置在打开柜子。 |
| `FoodCleanup` | 2 | 否 | 1. 拿起  [*食物物品*] 从台面并放置在柜子；2. 关闭柜子。 |

## 食物解冻

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `DefrostByCategory` | 4 | 否 | 1. there   mixed pile  frozen fruits 并 vegetables 在台面；2. locate 所有  frozen vegetables 并放置物品在碗在台面；3. 拿起所有  frozen fruits 并 defrost  在  running 水槽。 |
| `MicrowaveThawing` | 4 | 是 | 1. 从台面拿起食物并放入微波炉；2. 打开微波炉开始解冻。 |
| `MicrowaveThawingFridge` | 5 | 是 | 1. transport  肉类从冰箱到微波炉并打开微波炉。 |
| `MoveToCounter` | 3 | 是 | 1. 拿起  [*frosted 物品*] 从冰箱并放置在盘子在台面。 |
| `QuickThaw` | 2 | 是 | 1. retrieve  frozen 肉类并放置在锅在炉头；2. 打开炉头。 |
| `ThawInSink` | 2 | 否 | 1. 拿起物品从台面并放置在水槽；2. 打开水龙头。 |

## 装填上菜容器

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BuildAppetizerPlate` | 11 | 是 | 1. 拿起  [*cheese*]， 肉类， 并蔬菜从冰箱并放置在盘子在餐台到 create  appetizer 盘子。 |
| `DisplayMeatVariety` | 11 | 是 | 1. retrieve  chicken， shrimp， 并 steak 从冰箱并放置在  serving tray 在餐台。 |
| `MeatSkewerAssembly` | 2 | 否 | 1. 移动  kebab skewers 从盘子到烤箱 tray 用于烤箱 preparation。 |
| `MixedFruitPlatter` | 7 | 是 | 1. 拿起  [*fruit 物品*] 从冰箱并放置 one fruit 在 each 盘子在餐台。 |

## 煎制

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AssembleCookingArray` | 4 | 是 | 1. 移动肉类 onto  平底锅在炉灶；2. 移动  [*condiment*] 并蔬菜从柜子到台面 where  盘子。 |
| `DistributeSteakOnPans` | 2 | 是 | 1. 打开炉头用于  empty 平底锅；2. 移动 one steak 从  full 平底锅到  empty 平底锅到 achieve better 加热 distribution 并 more room 用于 maneuverability。 |
| `FryingPanAdjustment` | 2 | 否 | 1. 拿起并放置平底锅从  current 炉头到 another 炉头并打开炉头。 |
| `MealPrepStaging` | 4 | 是 | 1. 放置两个 pans onto different burners；2. 放置蔬菜并肉类在 different pans。 |
| `PressChicken` | 2 | 是 | 1. 拿起  spatula 从盘子并 lightly 按下  against  chicken 用于  few seconds；2. release。 |
| `RotatePan` | 2 | 否 | 1. rotate  平底锅到 help distribute  加热 more evenly。 |
| `SearingMeat` | 3 | 是 | 1. grab  平底锅从柜子并放置在  [*炉头 location*] 炉头在炉灶；2. 放置肉类在炉灶并打开炉头。 |
| `SetupFrying` | 2 | 是 | 1. 拿起平底锅从柜子并放置在炉灶；2. 打开炉头用于平底锅。 |

## 菜品装饰

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AddLemonToFish` | 4 | 是 | 1. 拿起  lemon 从冰箱并放置在盘子带有 fish 在餐台。 |
| `AddSugarCubes` | 2 | 否 | 1. 拿起  two sugar cubes 并放置在盘子带有  cake。 |
| `GarnishCake` | 2 | 是 | 1. 放置 one cherry either 在 top   cake or 在  cake 盘子， next 到  cake；2. 放置 one strawberry 在  cake 盘子， next 到  cake。 |
| `GarnishCupcake` | 4 | 是 | 1. 拿起  cinnamon bottle 从柜子并放置在  dining table， next 到盘子带有  cupcake；2. 拿起  chocolate 从餐台并放置在  cupcake 盘子。 |
| `GarnishPancake` | 4 | 是 | 1. 拿起  strawberry 从冰箱并放置在 top   pancake， located 在餐台。 |

## 装载洗碗机

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `LoadDishwasher` | 3 | 否 | 1. 从台面拿起餐具；2. 把餐具放入洗碗机；3. 关闭洗碗机门。 |
| `PrepareDishwasher` | 2 | 是 | 1. 打开洗碗机门；2. 拉出上层架子。 |

## 冰箱收纳

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `CreateChildFriendlyFridge` | 11 | 是 | 1. 放置  alcohol 在  top 搁架冰箱到 keep out  children's reach， 并放置  snack(s) 在 lower shelves 用于 easy access。 |
| `LoadCondimentsInFridge` | 9 | 是 | 1. 放置  [*condiments*] 从台面到  top 搁架冰箱；2. 如果  existing 物品在冰箱在  top 搁架， 移动到 other shelves。 |
| `LoadFridgeByType` | 7 | 是 | 1. 把装有蔬菜的碗放到冰箱指定搁架；2. 把装有肉类的碗放到另一个指定搁架。 |
| `LoadFridgeFifo` | 4 | 是 | 1. 放置肉类在台面 behind  older 肉类在  same 搁架冰箱到 maintain fifo order；2. 关闭冰箱门 when 完成。 |
| `LoadPreparedFood` | 4 | 是 | 1. 放置  tupperware 在  *top/second highest* 搁架冰箱；2. make space 用于  tupperware among  existing 物品通过 rearranging  在  shelves 如果 needed；3. 关闭冰箱门 when 完成。 |
| `MoveFreezerToFridge` | 2 | 是 | 1. 拿起  frozen 肉类从冷冻柜， 放置在任意搁架在冰箱， 并关闭冰箱并冷冻柜 doors when 完成。 |
| `PlaceVeggiesInDrawer` | 8 | 是 | 1. 放置蔬菜 inside  冰箱抽屉；2. 关闭抽屉 after placing  vegetables。 |
| `RearrangeFridgeItems` | 3 | 是 | 1. 把物品从初始搁架移动到目标搁架以腾出空间；2. 把台面上的水壶放到腾空的初始搁架。 |

## 制作果汁

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ChooseRipeFruit` | 2 | 否 | 1. 放置  non-rotten fruit 在  blender。 |
| `FillBlenderJug` | 2 | 是 | 1. 打开水龙头并放置  jug under  水用于  few seconds 到 fill  up。 |
| `JuiceFruitReamer` | 1 | 否 | 1. juice  水果通过 pressing  against  reamer。 |

## 制作沙拉

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `PrepareCheeseStation` | 4 | 是 | 1. 拿起  cheese 从冰箱并  cheese grater 从柜子并放置  next 到  salad 碗。 |
| `WashLettuce` | 2 | 否 | 1. 清洗  lettuce 在水槽通过 running 水 over。 |

## 制作奶昔/冰沙

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AddIceCubes` | 2 | 否 | 1. 放置  *two/three* ice cubes 在  blender。 |
| `AddSweetener` | 2 | 否 | 1. 放置  *two/three* sugar cubes 在  blender。 |
| `BlendIngredients` | 4 | 是 | 1. 打开  blender 盖子， 放置  fruit 在  blender， 关闭盖子， 并启动  blender。 |
| `PlaceStraw` | 2 | 是 | 1. 拿起  straw 从  *left/right* 抽屉并放置  inside  smoothie 杯子。 |
| `PrepareSmoothie` | 7 | 是 | 1. 拿起  [*fruits*] 从冰箱并放置在盘子 next 到  blender 到 make  fruit smoothie。 |

## 泡茶

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ArrangeTeaAccompaniments` | 4 | 是 | 1. 拿起 out  milk 从冰箱并放置在  tray 在台面；2. 拿起  sugar cube near  tray 并放置在  tray。 |
| `ServeTea` | 3 | 是 | 1. 拿起杯子  tea， navigate 到  dining table， 并放置在  saucer。 |
| `StrainerSetup` | 2 | 是 | 1. 打开柜子并 setup  tea strainer 通过 placing  在碗 upright。 |

## 制作吐司

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BreadSelection` | 2 | 是 | 1. 从  different types  pastries 在台面， select  croissant 并放置在砧板；2. retrieve  jar  jam 从柜子并放置  alongside  croissant 在砧板。 |
| `PrepareToast` | 3 | 否 | 1. 拿起  bread， 放置在砧板， 拿起  jam， 放置在台面， 并关闭柜子。 |
| `SweetSavoryToastSetup` | 4 | 是 | 1. 拿起  avocado 并 bread 从台面并放置在盘子；2. 拿起  jam 从柜子并放置  next 到盘子；3. 关闭柜子门。 |

## 冷冻柜空间管理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ClearFreezer` | 11 | 是 | 1. 拿起  [*食物物品*] 从冷冻柜并放置在碗在台面。 |
| `FreezeBottledWaters` | 8 | 是 | 1. 拿起  bottled waters 从台面并放置在冷冻柜到 freeze；2. 关闭冷冻柜门 after placing  bottles。 |
| `FreezeIceTray` | 4 | 是 | 1. ice cube tray  been filled 带有水；2. 拿起  tray 并放置在冷冻柜到 make ice；3. 关闭冷冻柜门 after placing  tray。 |
| `MaximizeFreezerSpace` | 3 | 是 | 1. 拿起  [*source 物品*] 从  *highest/second highest* 架子并移动到  *second highest/highest* 架子 where  [*target 物品*]  located， 到 maximize 冷冻柜 space；2. 关闭冷冻柜门 after rearranging。 |
| `MoveFridgeToFreezer` | 2 | 是 | 1. there's  肉类在冰箱 that needs 到  moved 到冷冻柜用于 freezing；2. 拿起肉类从冰箱， 放置在任意搁架在冷冻柜， 并关闭冰箱并冷冻柜 doors when 完成。 |
| `MoveToFreezerDrawer` | 2 | 是 | 1. 拿起  [*食物物品*] 从冰箱搁架并放置在冷冻柜抽屉到 begin freezing。 |
| `ReorganizeFrozenVegetables` | 3 | 是 | 1. 拿起  [*frozen vegetables*] out   冷冻柜并放置在  tupperware 容器在台面；2. 放置  tupperware containing  vegetables back 在冷冻柜。 |
| `SeparateFreezerRack` | 7 | 是 | 1. 拿起肉类容器 that   肉类并放置在  second highest 架子冷冻柜；2. 拿起蔬菜容器 that   蔬菜并放置在  highest 架子冷冻柜。 |

## 食材计量

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ChooseMeasuringCup` | 2 | 否 | 1. 打开  *left/right* 抽屉并放置  *larger/smaller* measuring 杯子在台面。 |
| `OrganizeMeasuringCups` | 3 | 否 | 1. 整理  measuring cups left 到 right， smallest 到 largest；2. measuring cups must  rotated so  handle points directly outward towards  front   台面；3. make sure  measuring cups  upright。 |
| `WeighIngredients` | 2 | 否 | 1. 拿起  [*packaged 食物*] 并放置在  digital scale 用于 weighing， 并关闭柜子。 |

## 肉类准备

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `PrepForTenderizing` | 2 | 是 | 1. 打开柜子， retrieve  rolling pin， 并放置  next 到肉类在砧板到准备用于 tenderizing。 |
| `PrepMarinatingMeat` | 2 | 是 | 1. 拿起肉类从  [*容器*] 并放置在砧板；2. 拿起  [*condiment*] 从柜子并放置  next 到砧板。 |

## 微波加热食物

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `FilterMicrowavableItem` | 6 | 是 | 1. 移除水果从碗并放置在  small 盘子；2. 放置碗带有 only  肉类在微波炉， 关闭门， 并按下启动按钮到微波炉肉类。 |
| `MicrowaveCorrectMeal` | 5 | 是 | 1. 放置碗 containing  食物在微波炉；2. 关闭微波炉门并按下启动按钮到微波炉  meal。 |
| `MicrowaveDefrostMeat` | 5 | 是 | 1. 放置  tupperware containing  frozen 肉类在微波炉；2. 关闭微波炉门并按下启动按钮到 defrost  肉类。 |
| `PlaceMicrowaveSafeItem` | 3 | 是 | 1. 放置  microwavable 物品在微波炉；2. 然后， 关闭微波炉门并按下启动按钮。 |
| `ReheatMeal` | 5 | 是 | 1. 把装有肉类和蔬菜餐食的保鲜盒放入微波炉；2. 关闭微波炉门；3. 按下启动按钮重新加热。 |
| `ReturnHeatedFood` | 4 | 是 | 1. 微波炉完成 heating  肉类并蔬菜；2. 拿起碗 containing  食物并放置在餐台。 |

## 混合食材

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BlendSalsaMix` | 6 | 是 | 1. 打开  blender 盖子并放置  tomato， chili pepper， 并 onion 在  blender；2. 放置盖子 back 在并按下  power 按钮到 make salsa。 |
| `BlendVegetableSauce` | 6 | 是 | 1. 蔬菜完成 being boiled；2. 关闭炉头用于  saucepan；3. 放置蔬菜在  blender 并打开  blender 到 make  蔬菜 sauce。 |
| `CheeseMixing` | 2 | 是 | 1. 放置  cheese 在锅 filled 带有 milk 并 stir。 |
| `MakeCheesecakeFilling` | 5 | 是 | 1. 加入  butter stick， sugar cube， 并 cream cheese stick 到  stand mixer 碗并；2. turn  speed knob 到 begin making cheesecake filling。 |
| `MakeChocolateMilk` | 2 | 否 | 1. 放置 one chocolate piece 并 one sugar cube 在  warm glass  milk 到 make chocolate flavored milk。 |
| `PrepareVeggieDip` | 9 | 是 | 1. 拿起蔬菜并  cream cheese 从冰箱， 放置在  blender， 并打开。 |

## 搅拌与混合

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ColorfulSalsa` | 4 | 否 | 1. 放置  avocado， onion， tomato， 并 bell pepper 在砧板。 |
| `MakeBananaMilkshake` | 11 | 是 | 1. grab  honey bottle 并放置  next 到  blender；2. grab  milk 并 banana 从冰箱， 并放置  milk next 到  blender while placing  banana 在  blender jug 到准备 making  banana milkshake。 |
| `SpicyMarinade` | 5 | 否 | 1. 打开柜子；2. 放置碗并 condiment 在台面；3. 放置  lime 并 garlic 在砧板。 |

## 餐具与容器整理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `EmptyDishRack` | 2 | 是 | 1. 拿起马克杯从碗碟架并放置  inside  打开柜子 where 所有 mugs go；2. 关闭柜子门(s)。 |
| `OrganizeMugsByHandle` | 2 | 是 | 1. 拿起马克杯在台面并放置在柜子带有  handle facing  right， like  mugs already inside。 |
| `StackBowlsCabinet` | 2 | 是 | 1. 拿起  bowls 在台面并 stack  在 top  one another 在打开柜子；2. 放置  smaller 碗在 top   larger 碗。 |

## 回收物整理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `RecycleBottlesBySize` | 7 | 是 | 1. 放置  large bottle 从餐台 next 到  other large bottle outside  水槽， 并放置  small bottle 从餐台 next 到  other small bottle inside  水槽。 |
| `RecycleBottlesByType` | 3 | 是 | 1. 移动  plastic bottles 在  middle 到  plastics group， 并  glass bottles 在  middle 到  glass group。 |
| `RecycleSodaCans` | 9 | 是 | 1. four empty soda cans  scattered around  kitchen；2. gather 所有  cans 并 cluster  next 到炉灶用于 recycling。 |
| `RecycleStackedYogurt` | 2 | 否 | 1. stack  yogurt containers 通过 placing  smaller containers 在 top   larger containers。 |

## 餐具整理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ArrangeUtensilsByType` | 4 | 是 | 1. 整理  utensils 通过 type；2. 拿起  wooden spoons 从台面并放置在打开柜子；3. 拿起  metallic utensils 从台面并放置在打开抽屉。 |
| `ClusterUtensilsInDrawer` | 2 | 是 | 1. 拿起  fork 从台面并放置在  *left/right* side   抽屉；2. 然后， 拿起  spoon 从台面并放置在  *right/left* side   抽屉；3. make sure that  utensils  clustered 通过 type。 |
| `OrganizeMetallicUtensils` | 7 | 是 | 1. 拿起  metallic utensils 从台面并放置在打开抽屉；2. make sure not 到拿起任意 wooden utensils。 |

## 打包午餐

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `PackFoodByTemp` | 15 | 是 | 1. 放置  cold 物品 ([*cold 物品*]) 从冰箱在 one tupperware 在餐台；2. 放置  warm 物品 ([*warm 物品*]) 在炉灶 area 在 another tupperware。 |
| `PackFruitContainer` | 7 | 是 | 1. 拿起  [*fruits*] 从冰箱并放置  inside  [*容器*] 在台面。 |
| `PackIdenticalLunches` | 15 | 是 | 1. 放置 one 蔬菜并 one 肉类在 each tupperware 在  nearby 台面， 到 pack two identical lunches。 |
| `PackSnack` | 3 | 是 | 1. 拿起  snack 从柜子并放置  next 到  tupperware containing  packed lunch 在台面到拿起在  go。 |

## 装盘

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BalancedMealPrep` | 2 | 是 | 1. 放置肉类在盘子并放置蔬菜在  same 盘子。 |
| `PlateSteakMeal` | 2 | 是 | 1. create  steak meal 通过 picking up  steak 并 asparagus 从  tray 并 placing  在盘子；2. leave  [*other 食物*] 在  tray。 |
| `PlateStoreDinner` | 8 | 是 | 1. 放置 one   steaks 在盘子并  other 在碗；2. 放置碗在冰箱。 |

## 餐食分装

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `DistributeChicken` | 7 | 是 | 1. 拿起  chicken drumsticks 从平底锅并放置 one 在 each 盘子在餐台。 |
| `PortionFruitBowl` | 4 | 是 | 1. 拿起  fruits 从盘子在餐台并放置 two fruits 在 each 碗。 |
| `PortionHotDogs` | 4 | 是 | 1. 放置 one bun 并 one sausage 从碗在 each 盘子。 |
| `PortionInTupperware` | 4 | 是 | 1. 放置 one cooked 食物并 one 蔬菜从  large 盘子到 each tupperware 容器。 |
| `PortionOnSize` | 7 | 是 | 1. 拿起  smaller 碗并放置  next 到  appetizer 盘子在餐台， 并放置  larger 碗 next 到盘子带有 cooked 食物。 |
| `PortionYogurt` | 4 | 是 | 1. 放置 two yogurt containers 在 each 盘子。 |
| `ScalePortioning` | 6 | 是 | 1. 拿起肉类从冰箱并放置在  digital scale 在台面通过冰箱；2. wait  few seconds 用于  reading；3. 移动到盘子在餐台。 |

## 准备热巧克力

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AddMarshmallow` | 3 | 是 | 1. 拿起  marshmallow 并放置在杯子  hot chocolate 在餐台。 |
| `SweetenHotChocolate` | 1 | 是 | 1. 拿起  sugar cube(s) 并放置在  saucepan 带有 hot chocolate。 |

## 准备腌料

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BlendMarinade` | 9 | 是 | 1. grab  蔬菜从冰箱， 放置在  blender， 关闭盖子， 并打开  blender 到 create  marinade。 |
| `GatherMarinadeIngredients` | 6 | 是 | 1. retrieve  bottle 并  shaker 从柜子并放置  next 到  mixing 碗在台面；2. 加入 garlic 从冰箱到  mixing 碗。 |
| `PlaceMeatInMarinade` | 5 | 是 | 1. retrieve  肉类从冰箱并放置在  saucepan 带有 marinade 在炉灶。 |

## 准备三明治

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `GatherVegetables` | 2 | 否 | 1. 拿起蔬菜从水槽并放置在砧板在台面。 |
| `HeatKebabSandwich` | 6 | 是 | 1. 拿起  kebab skewer 并 baguette bread， 并放置  inside  烤面包箱；2. 关闭烤面包箱门并启动通过 setting  timer。 |
| `HotDogSetup` | 12 | 是 | 1. 移动  hot dog bun 到盘子在  dining table 并放置  [*condiment*] next 到 that 盘子；2. navigate 到冰箱并拿起  sausage 并放置在盘子 as well。 |
| `PrepareSandwichStation` | 7 | 是 | 1. carry  碗 containing  sandwich ingredients 从冰箱并放置在台面 near  烤面包箱；2. next， 拿起  baguette 从冰箱并放置  near  烤面包箱 as well。 |
| `PrepareSausageCheese` | 8 | 是 | 1. retrieve  sausage 并 cheese 从冰箱并放置在砧板在台面。 |
| `ToastHeatableIngredients` | 4 | 否 | 1. 打开烤面包箱门并放置  heatable sandwich ingredients 在架子 only；2. make sure  cold/unheable 物品  still 在盘子。 |

## 重新加热食物

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `HeatMug` | 4 | 是 | 1. 拿起马克杯从台面并放置  inside  微波炉；2. 关闭微波炉。 |
| `MakeLoadedPotato` | 6 | 是 | 1. retrieve  reheated potato 从微波炉；2. 放置  potato， cheese， 并 condiment 在砧板。 |
| `ReheatMeatOnStove` | 3 | 是 | 1. cooked 肉类  gone cold；2. 放置在平底锅在炉灶到重新加热。 |
| `SimmeringSauce` | 4 | 是 | 1. 放置平底锅在  [*炉头 location*] 炉头在炉灶；2. 放置  tomato 并  onion 在平底锅并打开  [*炉头 location*] 炉头。 |
| `WaffleReheat` | 4 | 是 | 1. 打开微波炉；2. 把装有华夫饼的碗放入微波炉；3. 关闭微波炉门；4. 打开微波炉。 |
| `WarmCroissant` | 2 | 是 | 1. 拿起  croissant 并放置在平底锅；2. 打开炉灶到 warm  croissant。 |

## 补充物资

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `FreshProduceOrganization` | 4 | 是 | 1. 放置  [*fruit/蔬菜*] 在  correct 冰箱架子 based 在 whether    fruit or 蔬菜；2. finally， 关闭冰箱门。 |
| `RefillCondimentStation` | 4 | 是 | 1. dining table condiment station  running low 在 [*condiment*]；2. refill  condiment station 通过 opening  柜子并 taking  [*condiment*] 从柜子并 placing  在  dining table。 |
| `RestockCannedFood` | 2 | 是 | 1. restock  canned 食物柜子通过 taking  new fresh canned 食物从台面并 placing  在柜子；2. 然后， 拿起  old canned 食物从柜子并放置在台面。 |
| `RestockSinkSupplies` | 2 | 是 | 1. restock  水槽 supplies 通过 placing  sponge 并 soap dispenser 在台面并 next 到水槽；2. make sure  soap dispenser  upright。 |

## 砧板消毒

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `RemoveCuttingBoardItems` | 2 | 否 | 1. 移除蔬菜并肉类从砧板到准备用于 cleaning。 |
| `RinseCuttingBoard` | 2 | 否 | 1. 打开水龙头带有 hot 水并冲洗砧板用于 ten seconds；2. 关闭  faucet；3. push  砧板到 reach  水如果 necessary。 |
| `SanitizePrepCuttingBoard` | 5 | 是 | 1. grab  spray bottle 从柜子；2. 放置  next 到砧板用于 sanitizing；3. grab  dish brush 从  countertop next 到水槽并放置  next 到砧板。 |
| `ScrubCuttingBoard` | 2 | 是 | 1. 拿起  sponge 从台面并清洁砧板通过 briefly scrubbing or pressing down 在砧板；2. 一旦完成， release  sponge。 |

## 台面消毒

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ArrangeSinkSanitization` | 7 | 是 | 1. 放置  sponge 并 dishbrush next 到水槽在台面。 |
| `CleanMicrowave` | 5 | 是 | 1. 放置  sponge inside  微波炉到准备用于 sanitizing。 |
| `CountertopCleanup` | 5 | 是 | 1. 把餐具放入打开的抽屉；2. 把餐具或碗盘放入打开的柜子；3. 把装有食物的碗放入冰箱。 |
| `PrepForSanitizing` | 3 | 否 | 1. 拿起  [*cleaning supplies*] 从柜子并放置在台面。 |
| `SanitizeSink` | 2 | 是 | 1. 拿起  sponge 从台面并 rub  across  水槽；2. when 完成， drop  sponge。 |
| `WipeTable` | 2 | 否 | 1. 拿起  sponge 从台面并 wipe  台面通过 gripping  并 pressing  against  surface；2. 一旦完成， release  sponge。 |

## 炒蔬菜

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AdjustHeat` | 3 | 是 | 1. 放置蔬菜在平底锅；2. 打开  [*炉头 location*] 炉头并设置加热到 medium。 |
| `ButterOnPan` | 2 | 是 | 1. 拿起  butter stick 并放置在平底锅；2. 打开炉头用于平底锅。 |
| `PlaceVegetablesEvenly` | 2 | 否 | 1. 拿起蔬菜从台面并放置  inside  平底锅；2. make sure they  evenly distributed 并 not overlapping each other。 |
| `PreheatPot` | 2 | 是 | 1. 打开炉头加热锅；2. 等待 30 秒；3. 把蔬菜加入锅中。 |
| `ShakePan` | 2 | 否 | 1. grip  平底锅 handle， lift 并 shake  lightly 到移动蔬菜 inside  平底锅；2. 放置平底锅 back down 在炉灶一旦完成。 |
| `StirVegetables` | 4 | 是 | 1. 放置蔬菜在锅；2. retrieve  spatula 并 lightly stir  vegetables 在锅。 |
| `TiltPan` | 2 | 否 | 1. tilt  平底锅 slightly 到移动蔬菜 inside  平底锅；2. 放置平底锅 back down 在炉灶一旦完成。 |

## 食物调味

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `LemonSeasoningFish` | 2 | 是 | 1. retrieve  lemon 从冰箱并放置在砧板带有  fish。 |
| `SeasoningSteak` | 3 | 是 | 1. retrieve  shaker 从柜子并放置  next  steak 在盘子在  dining table。 |
| `SetUpSpiceStation` | 11 | 是 | 1. collect  [*spice*]， [*bottle*]， 并 [*shaker*] 从柜子并放置  next 到炉灶。 |

## 供应饮品

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AlcoholServingPrep` | 8 | 是 | 1. 打开柜子 directly 在 front；2. 移动  [*alcohol*] 并  [*杯子*] 到餐台带有  [*decoration*] 在。 |
| `DeliverStraw` | 4 | 是 | 1. 拿起  straw 从抽屉在 front 并放置  inside  glass 杯子在餐台。 |
| `MatchCupAndDrink` | 7 | 是 | 1. there   wine glass 并 glass 杯子在餐台；2. 放置  wine bottle next 到  wine glass 并  juice bottle next 到  glass 杯子。 |
| `PrepareCocktailStation` | 12 | 是 | 1. there   碗在餐台；2. grab  lemon wedge 从冰箱并放置在碗；3. get  liquor bottle 并 glass 杯子从柜子并放置  next 到碗。 |
| `PrepareDrinkStation` | 11 | 是 | 1. there   tray 在餐台；2. grab  杯子， 马克杯， 并 pitcher 并放置在  tray；3. pitcher may  placed next 到  tray 如果  cannot fit  tray。 |
| `ServeMealJuice` | 7 | 是 | 1. 拿起  orange juice cups 从台面并放置 one next 到 each 盘子在餐台。 |
| `SetupSodaBowl` | 8 | 是 | 1. 打开冰箱， grab two sodas， 并放置两个 sodas inside  碗带有 ice 在餐台。 |

## 上菜

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `DessertUpgrade` | 2 | 是 | 1. 移动  dessert 物品从  plates 到  tray。 |
| `PanTransfer` | 3 | 否 | 1. 拿起平底锅并 dump  vegetables 在  onto  盘子；2. return  平底锅到炉灶。 |
| `PlaceFoodInBowls` | 4 | 是 | 1. 拿起两个 bowls 并放置在台面；2. 放置  [*食物1*] 在 one 碗并放置  [*食物2*] 在  other 碗。 |
| `PrepareSoupServing` | 3 | 是 | 1. 打开柜子并移动  ladle 到锅；2. 关闭柜子。 |
| `ServeSteak` | 4 | 是 | 1. 拿起平底锅带有  steak 在并放置在  dining table；2. 放置  steak 在盘子。 |

## 摆餐桌

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AlignSilverware` | 7 | 是 | 1. 拿起  fork 并放置在  left side   盘子；2. 拿起  spoon 并放置在  right side   盘子；3. make sure they're directly next 到盘子。 |
| `ArrangeBreadBasket` | 5 | 是 | 1. 打开柜子， 拿起面包从柜子并放置在  basket；2. 移动  basket 到餐台。 |
| `ArrangeBreadBowl` | 5 | 是 | 1. 拿起  heated 面包从烤面包箱并放置在碗；2. 移动  bread 碗到餐台。 |
| `ArrangeDrinkware` | 7 | 是 | 1. 拿取水壶和杯子；2. 把它们摆放到餐台用于饮品服务。 |
| `BeverageOrganization` | 7 | 是 | 1. 移动  drinks 到餐台。 |
| `DateNight` | 8 | 是 | 1. get  [*decoration*] 并  [*alcohol*] 从柜子并移动到餐台。 |
| `SeasoningSpiceSetup` | 8 | 是 | 1. 移动  [*condiments*] 从柜子到餐台。 |
| `SetBowlsForSoup` | 8 | 是 | 1. 从柜子拿取碗；2. 把碗放到餐桌上的盘子旁用于盛汤。 |
| `SetupBowls` | 7 | 是 | 1. 拿起  bowls 从柜子并放置 each 碗在 front   stool 在餐台。 |
| `SetupButterPlate` | 8 | 是 | 1. retrieve  butter 从冰箱并放置在盘子在  dining table；2. retrieve  butter knife 通过冰箱并放置  near  盘子。 |
| `SetupFruitBowl` | 8 | 是 | 1. retrieve  [*fruits*] 从冰箱并放置放入碗在  dining table。 |
| `SetupWineGlasses` | 7 | 是 | 1. retrieve  wine glasses 从台面并放置 one next 到 each 盘子。 |
| `SizeSorting` | 1 | 否 | 1. stack  [*tableware*] 从 largest 到 smallest。 |

## 炖煮酱汁

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `TurnOffSimmeredSauceHeat` | 1 | 是 | 1. wait until  marinara sauce  simmered 到  deep red color 并；2. 关闭炉灶。 |

## 肉类切配

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `CleanBoard` | 7 | 是 | 1. 把砧板和刀移动到水槽；2. 把喷雾瓶从柜子移动到台面。 |
| `RetrieveMeat` | 3 | 是 | 1. retrieve  肉类从冰箱并放置在砧板。 |
| `SetUpCuttingStation` | 2 | 是 | 1. 拿起  knife 从抽屉并放置在砧板；2. 放置肉类从盘子到砧板。 |

## 慢炖

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AddToSoupPot` | 4 | 是 | 1. 移除盖子从  simmering saucepan 并加入蔬菜从盘子到  saucepan；2. 放置盖子 back 到 continue slow cooking  soup。 |
| `BeginSlowCooking` | 2 | 是 | 1. 设置炉头 covering  saucepan 带有 broth 到 low 加热， 并放置盖子在  saucepan 到 begin slow cooking。 |
| `StopSlowCooking` | 2 | 是 | 1. 打开 off  炉头并放置  saucepan 盖子台面到准备 serving  broth。 |

## 准备零食

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `BreadAndCheese` | 2 | 否 | 1. 放置  bread 并 cheese 在砧板。 |
| `CerealAndBowl` | 4 | 否 | 1. 打开柜子；2. 拿起  cereal 并碗从柜子并放置在台面 next 到  milk；3. 关闭柜子。 |
| `MakeFruitBowl` | 3 | 否 | 1. 打开柜子；2. 拿起  [*fruits*] 从柜子并放置放入碗；3. 关闭柜子。 |
| `VeggieDipPrep` | 3 | 是 | 1. 放置  two vegetables 并碗 onto  tray 用于 setting up  蔬菜 dip station。 |
| `YogurtDelightPrep` | 2 | 否 | 1. 放置  yogurt 并 fruit(s) onto  台面。 |

## 食材分类

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `SeparateRawIngredients` | 11 | 是 | 1. 放置肉类/seafood 物品(s) 在砧板并蔬菜物品(s) 在碗到 avoid cross-contamination。 |
| `SortBreakfastIngredients` | 7 | 是 | 1. 放置  jam next 到盘子  bread 并  cereal next 到  carton  milk 在餐台。 |

## 蒸食物

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `MultistepSteaming` | 7 | 是 | 1. 打开水龙头；2. 移动蔬菜从台面到水槽；3. 关闭水槽；4. 移动蔬菜从水槽到锅 next 到炉灶；5. 移动锅到  [*炉头 location*] 炉头。 |
| `SteamFish` | 2 | 是 | 1. 放置  fish 在平底锅带有 soy sauce 并打开炉灶 knob 到启动 steaming。 |
| `SteamInMicrowave` | 6 | 是 | 1. 从水槽拿起蔬菜并放入碗；2. 拿起碗并放入微波炉；3. 关闭微波炉门；4. 按下启动按钮。 |

## 蒸蔬菜

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `PrepareVeggiesForSteaming` | 4 | 是 | 1. 冲洗蔬菜在  colander under running 水 briefly；2. 放置  colander 带有  vegetables 在台面 next 到炉灶到准备用于 steaming。 |
| `RemoveSteamedVegetables` | 2 | 是 | 1. 蔬菜完成 steaming；2. 拿起  out   saucepan 并放置在盘子在台面。 |
| `SteamVeggiesWithWater` | 3 | 是 | 1. 放置蔬菜在  saucepan 带有水并放置盖子在到 begin steaming  vegetables。 |

## 剩菜收纳

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `FreezeCookedFood` | 4 | 是 | 1. store  食物在冷冻柜用于 long-term storage；2. 关闭冷冻柜门 after placing  cooked 食物。 |
| `PrepareStoringLeftovers` | 7 | 是 | 1. retrieve  tupperware 并 aluminum foil 从柜子并放置在餐台 near  盘子带有食物到准备用于 storing leftovers。 |
| `StoreDumplings` | 11 | 是 | 1. 放置 two dumplings 放入 each   tupperware containers 并；2. 放置  containers 在冰箱。 |
| `StoreLeftoversByType` | 7 | 是 | 1. transport  肉类并蔬菜到冰箱；2. 放置肉类在架子 containing 肉类并放置蔬菜在架子 containing  蔬菜， ensuring meats 并 vegetables  在 different racks。 |
| `StoreLeftoversInBowl` | 5 | 是 | 1. 拿起  chicken drumstick 并蔬菜从 their plates 并放置在碗；2. 放置碗在冰箱。 |

## 柜子与抽屉整理

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `DrawerUtensilSort` | 3 | 是 | 1. 打开  *left/right* 抽屉并放置  utensils inside。 |
| `OrganizeCleaningSupplies` | 3 | 是 | 1. 打开柜子；2. 拿起  [*cleaning supply*] 并放置  next 到水槽；3. 关闭柜子。 |
| `PlaceBreakfastItemsAway` | 7 | 是 | 1. 拿起  [*breakfast 物品*] 从餐台并放置在打开柜子到 store away breakfast 物品。 |
| `SnackSorting` | 2 | 否 | 1. 放置  bar 在碗并关闭抽屉。 |
| `UtensilShuffle` | 2 | 是 | 1. 拿起并放置  utensils 放入打开抽屉。 |

## 烤面包

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `GetToastedBread` | 4 | 是 | 1. 启动烤面包机；2. 等压杆弹起；3. 把烤好的面包拿到餐台上的盘子。 |
| `PJSandwichPrep` | 5 | 是 | 1. 拿起  two toasted bread slices 并放置在碗 next 到烤面包机；2. carry  碗到餐台并放置  next 到  peanut butter bottle 并 jelly jar 到 make  peanut butter 并 jelly sandwich。 |
| `ServeWarmCroissant` | 3 | 否 | 1. 拿起 out  warm croissant 从烤面包箱并放置在盘子。 |
| `ToastBagel` | 5 | 否 | 1. 从盘子拿起贝果；2. 把贝果放到烤面包箱架子；3. 关闭门；4. 设置计时器。 |
| `ToastBaguette` | 5 | 否 | 1. 拉出烤面包箱架子 outwards；2. 拿起  baguette 从盘子并放置在烤面包箱架子；3. 关闭烤面包箱门 fully 并设置  temperature 到  lowest setting。 |
| `ToastOnCorrectRack` | 7 | 是 | 1. 打开烤面包箱门；2. 把面包放到指定上层或下层架子；3. 把肉类放到另一层架子。 |
| `ToastOneSlotPair` | 2 | 否 | 1. 移动  bread slice 从  broken *left/right* slot 到  working *right/left* slot-pair so 两个 slices can 烘烤 together；2. 按下 that lever 到启动 toasting。 |

## 洗餐具

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `ClearSink` | 4 | 是 | 1. 从水槽移除食物；2. 把台面上的餐具移动到水槽用于清洗。 |
| `CollectWashingSupplies` | 2 | 是 | 1. locate 并 retrieve  [*cleaning 物品*] 从柜子；2. 放置  next 到水槽。 |
| `DivideBasins` | 1 | 是 | 1. 移动任意 existing 物品从  left basin 到  right basin 用于 drying；2. 移动  [*washing 物品*] 从台面到  left basin   水槽用于 washing。 |
| `DryDishes` | 2 | 否 | 1. 拿起杯子并碗从水槽并放置在台面用于 drying。 |
| `DryDrinkware` | 2 | 否 | 1. 拿起马克杯并放置  upside down 在打开柜子。 |
| `DumpLeftovers` | 3 | 是 | 1. dump  剩菜从碗 so  碗 can  washed；2. 放置碗在水槽 after clearing  leftovers。 |
| `PlaceDishesBySink` | 7 | 是 | 1. 拿起餐具从台面并放置在台面 next 到水槽。 |
| `PlaceOnDishRack` | 2 | 是 | 1. transfer  清洁物品从  right 水槽 basin 到碗碟架用于 drying；2. leave 任意 dirty 物品在  left basin untouched。 |
| `PreRinseStation` | 2 | 是 | 1. 放置  [*left object*] 在  left basin 并  [*right 物品*] 在  right basin   水槽。 |
| `PreSoakPan` | 3 | 否 | 1. 拿起平底锅并 sponge 并放置放入水槽；2. 打开水。 |
| `ReturnWashingSupplies` | 8 | 是 | 1. 拿起  [*cleaning supplies*] 从水槽， 并放置  back 在柜子。 |
| `RinseBowls` | 3 | 是 | 1. 冲洗 each 碗用于  few seconds 并放置  down 在水槽 or 在台面。 |
| `RinseFragileItem` | 4 | 是 | 1. 冲洗  teapot briefly 并放置在碗碟架 when 完成。 |
| `ScrubBowl` | 2 | 否 | 1. use  sponge 到 scrub  碗 inside  水槽。 |
| `SoakSponge` | 2 | 否 | 1. soak  sponge 带有水用于  few seconds 并；2. drop  inside  水槽。 |
| `SortingCleanup` | 5 | 是 | 1. 拿起马克杯并放置在水槽；2. 拿起碗并放置在打开柜子并；3. 关闭柜子。 |
| `StackBowlsInSink` | 2 | 否 | 1. stack  bowls 在水槽。 |
| `TransportCookware` | 7 | 是 | 1. 移动锅并平底锅从炉灶放入水槽。 |

## 清洗果蔬

| 任务名 | 子任务数 | 移动 | 阶段说明 |
|---|---:|:---:|---|
| `AfterwashSorting` | 4 | 否 | 1. 拿起  [*蔬菜1*] 从水槽并放置在 one 碗；2. 放置  [*蔬菜2*] 在  other 碗；3. 关闭水龙头。 |
| `AirDryFruit` | 3 | 否 | 1. 拿起水果从水槽并放置在砧板到 airdry；2. 关闭水龙头。 |
| `ClearClutter` | 6 | 是 | 1. 拿起所有 produce 物品并放置在水槽；2. 打开水龙头到清洗；3. 关闭水槽并放置在  tray。 |
| `ClearSinkSpace` | 2 | 否 | 1. 从水槽拿起物品并放到台面，清出用于清洗农产品的水槽空间。 |
| `DrainVeggies` | 4 | 是 | 1. dump  蔬菜从锅放入水槽；2. 打开水并清洗蔬菜；3. 关闭水并放置蔬菜 back 在锅。 |
| `GatherProduceWashing` | 7 | 是 | 1. 拿起食物从冰箱并放置在碗 next 到水槽。 |
| `PrepareVegetableRoasting` | 4 | 是 | 1. 拿起蔬菜从冰箱并 hold  under  水龙头到清洗；2. 放置在  tray next 到水槽到准备用于 roasting。 |
| `PrewashFoodAssembly` | 5 | 是 | 1. 拿起食物从柜子并放置在碗；2. 拿起碗并放置在水槽；3. 打开水槽 facuet。 |
| `PrewashFoodSorting` | 8 | 是 | 1. 拿起  [*食物1*] 从柜子并放置在 one 碗；2. 拿起  [*食物2*] 从水槽并放置在  other 碗。 |
| `WashFruitColander` | 4 | 否 | 1. 放置  colander 在水槽， 放置水果在  colander， 并打开水龙头并 pour 水 over  colander。 |
| `WashInSaucepan` | 7 | 是 | 1. 拿起蔬菜从冰箱并放置在  saucepan 带有水 next 到水槽用于 soak washing。 |
