# 2022-IDPCD
2022년 2학기 개별연구(Independent Capstone Design)

## 연구 주제
- <b>노인을 위한 제스처인식 시스템 설계</b>
일반적인 컴퓨터 환경에 익숙하지 않고, 거동이 불편한 노인과 같은 사용자를 위하여 특화된 비전 기반 제스처 인식 시스템을 설계하고, 실험한다.

### Part 1. 개요
본 연구에서는 일반적인 컴퓨터 환경에 익숙하지 않고, 거동이 불편한 노인과 같은 사용자를 위하여 특화된 비전 기반 제스처 인식 시스템을 설계하고, 실험하고자 한다.

#### [연구 목표]
정전식 터치패드로 동작하는 기존의 키오스크는 시력 저하와 같은 신체적 노화, 기계에 대한 부정적 인식 등으로 인해 노인에게 불편함을 초래한다. 따라서 카메라를 통해 손동작(제스처)을 인식하여 동작하는 키오스크를 제작하기 위해 비접촉식 제스처 인식 시스템을 개발하고자 한다. 노인이나 시각장애인을 포함한 모든 사용자는 글자를 읽고 해당 글자를 터치하는 것 대신, 글자를 보거나 듣고 적절한 손동작을 취하는 것으로 의사 표현을 하여 보다 쉽게 키오스크를 이용할 수 있다.

#### [기대효과]
본 연구를 통해 노인, 시각장애인 등도 키오스크를 어려움 없이 사용할 수 있게 되어 디지털 격차가 해소되기를 기대한다.

### Part 2. 연구 배경
#### [문제 인식]
> 키오스크에 '쩔쩔'…노인들 "햄버거 먹기 힘드네"

출처 : https://www.hankyung.com/society/article/2022051694181

서울디지털재단이 16일 발표한 ‘서울시민 디지털역량 실태조사’ 결과에 따르면 서울에 사는 만 55세 이상 고령층의 디지털 기술 이용 수준은 43.1점(100점 만점)으로 서울 시민 전체 평균(64.1점) 대비 32.7% 낮았다. 키오스크 주문 시스템과 관련해 고령층의 54.2%는 ‘단 한 번도 사용해본 적 없다’고 응답했다. 나이가 들수록 키오스크 이용 경험은 줄어들어 75세 이상은 13.8%만이 키오스크를 사용한 경험이 있는 것으로 나타났다.
반면 키오스크의 비중은 커지고 있다. 주요 패스트푸드 기업인 K사는 2017년 전국 모든 매장에 키오스크를 도입한다고 선언한 뒤 1년 만인 2018년에 이를 달성했다. 200여개 매장을 운영하는 K사는 매장당 3∼4대의 키오스크를 두고 있다. K사 관계자는 "키오스크 100% 도입으로 접객 직원을 매장 위생관리나 조리 등으로 돌려 보다 효율적으로 인력을 운영할 수 있게 됐다"고 말했다.
> 실제 인터뷰

글씨가 너무 작아요. 글씨가 잘 안보입니다. 스마트폰처럼 확대해서 볼 수 있었으면 좋겠습니다. 화면을 가까이 들여다보아야 하니까 몸도 피곤하고, 주문 시간이 오래 걸리네요(참여자 2).

내가 원하는 정보를 얻을 수가 없어요. 나는 (음식) 주문할 때 인기 메뉴를 물어보는 편인데, 오스크에서는 그 정보를 얻을 수가 없잖아요. 사람들이 가장 많이 먹는 음식은 무엇인지, 추천 상품은 무엇인지 알아야하는데..... 나는요. 월래 선택을 잘 못해요. 그래서 식당가면 인기 메뉴만 먹어요(참여자 12).

나이가 드니까 행동이 느려지고, 둔해지네요. 정말 화면을 살짝만 만졌는데, 그것이 장바구니에 담겨서 문
이 되었더라고요. 하나만 시키려고 했는데, 나중에 보니까 두 개가 주문이 됐어요(참여자 15).

화면이 보여야지 뭘 하지요. 나이 먹어서 서 있는 것도 힘든데, 손도 움직여야 하니까 여간 힘든 것이 아니예요. 그리고 지문이 닳아서 그런 것인지 화면 만지는 것도 잘 안 돼요(참여자 12).

내가요. 못해도 60년 넘게 직원한테 (음식)주문을 하고 산 사람이예요. 그런데, 갑자기 기계 가져다놓고 그것을 이용하라고 하는데 거부감이 어떻게 안 들어요? 나는 솔직히 무섭습니다. 막말로 내가 기계 만지다가 고장이라도 났어. 나한테 수리비 내놓으라고 할 것 아니야(참여자 9).

영어를 모르는 사람은 이용하지 말라는 것이지. 영어가 왜 이렇게 많아. 그리고 영어만 색깔을 다르게해서 눈에 더 잘 띄게 해놨어. 메뉴도 글자만 한글이야. 다 영어고 그냥 영어를 한글로 써놓았더라고.... 사진보고 대충 선택하기는 했는데...... 결국은 포기하고 안 먹었어(참여자 13).

내가 살면 얼마나 산다고.... (키오스크) 이용 방법을 배우는 것이 얼마나 의미가 있겠어요?. 그 시간에 내가 하고싶은 일 하면서 사는 것이 훨씬 낫지 싶어요. 키오스크? 그것을 할 수 있다고 내 삶이 얼마나 달라질까요? 밖에 나가서 밥을 자주 사먹는 것도 아니고, 그리고 키오스크 없는 곳도 많잖아요. 그냥 키오스크 사용 안하는 식당 갈래요(참여자 13)

> 키오스크가 가진 문제점 : 비주얼 크라우딩

권오상 울산과학기술원(UNIST) 바이오메디컬공학과 교수는 “사람은 나이가 들면서 시각, 인지 기능이 자연스럽게 줄어든다”며 “그중 하나는 시각정보가 좁은 공간에 여러 개 있을 때 이를 명확하게 인지하지 못하는 ‘비주얼 크라우딩(시각적 혼잡)’”이라고 말했다. 미국 앨라배마대 연구팀에 따르면 고령층이 짧은 문장을 읽어내는 속도는 젊은층보다 약 30% 느리다. 글자를 인식하기 위해 필요한 자간도 31% 넓어야 한다. 고령층은 정보를 읽어내는 속도가 느리고, 밀집된 정보를 해석하기 어렵다는 의미다. 키오스크를 이용할 때는 더 큰 비주얼 크라우딩이 나타난다.
화면의 크기는 작으면서도 많은 정보가 담겨있기 때문이다. 글자 크기는 작고 자간은 좁아 화면을 읽어내는 데 느끼는 어려움은 더 커진다.

키오스크는 기본적으로 터치스크린 방식을 적용하여 업무를 처리한다. 글자를 보고 해석한 후 원하는 글자를 터치해야 한다. 따라서 노인과 시각장애인을 비롯하여 글자를 읽고 해석하는 것이 어려운 사람들은 키오스크를 다루는 것이 쉽지 않다.


#### [해결 방안]
> 정전식 터치패드가 아닌 카메라를 통해 손동작(제스처)을 인식하여 동작하는 키오스크를 제작

글자를 읽고 해석해 터치하는 것이 어렵다면, 글자를 읽어주고 제스처로 의사 표현을 하게끔 하면 된다. 마치 청각 장애인이 수화로 의사소통하듯, 우리는 키오스크와 제스처로 의사소통하는 것이다.

> [시각보다 청각으로 소통하는 것이 더 우수할 것이라는 가정의 근거](https://www.yna.co.kr/view/AKR20220524039900009)

이와 함께 노인성 황반변성과 유사한 시력 조건을 만들기 위해 한번은 고글을 쓰고 한번은 고글 없이 테스트를 받게 했다.

그 결과 글로 된 질문을 눈으로 보고 답을 쓰는 테스트에서는 고글을 썼을 때가 쓰지 않았을 때보다 성적이 훨씬 나쁘게 나왔다.

그러나 말로 하는 질문을 귀로 들으면서 하는 테스트 성적은 고글을 썼을 때나 쓰지 않았을 때나 별 차이가 없었다.

> 구체화 - 노인의 제스처

노인의 경우 전신으로 제스처를 취하기 보다는 손으로 제스처를 취하는 것이 쉬우며, 높이가 낮다. 따라서 키오스크 모니터 아래에 제스처 인식을 위한 카메라(영상 인식)가 있다고 가정하고, 의사 표현에 적합한 제스처 동작을 선정하였다.

선정을 위해 고려한 것은 사용자가 노인이라는 점이다.
1. 손떨림이 있으므로 확실히 크고 비교적 정확하게 취할 수 있는 제스처여야 할 것.
2. 한 손으로도 취할 수 있는 제스처여야 할 것.
3. 치매 예방에 좋은 손 협응체조를 참고할 것.

(근거) 보통 나이가 들면서 손놀림의 정확성이 떨어집니다. 아울러 여러 가지 뇌질환에 의해서 정교한 손놀림이 점차 어려워질 수도 있습니다. 퇴행성 뇌질환 중에선 대표적으로 파킨슨병이 손 움직임이 느리고, 작아지는 증상이 나타납니다. 이 때문에 글씨가 잘 써지지 않거나 단추를 채우는 등 세밀한 손동작에 어려움이 생깁니다.

기억력 등 인지기능 저하가 주요 증상인 알츠하이머병 노인은 손놀림의 정확도, 반응속도, 리듬 등이 건강한 노인보다 떨어집니다. 특히 양손을 모두 이용하는 동작이 많은 영향을 받습니다.

양손을 이용해서 하는 동작에는 여러 가지 뇌 부위가 관여합니다. 오른쪽‧왼쪽 뇌가 모두 잘 활동해야 하고, 양쪽 뇌를 연결하는 전선다발인 뇌량 역시 잘 기능해야 합니다. 때문에 손놀림의 정확성이 떨어지는 것이 점차 악화되면 뇌 질환의 유무 등을 확인을 하는 것이 필요합니다.
-강동경희대병원 신경과 이학영 교수-

위의 3가지를 고려하여 선정한 제스처는 '주먹 쥐기' / '아래와 위(good,bad)' / '숫자 접기(1,2,3)' 으로 총 6개이다.

해당 제스처들이 과연 노인이 하기 적절한 것에 대한 근거를 명확하게 제시하셔야 합니다. (노인의 신체적 특징에 대한 논문 혹은 설문조사 같은 자료들을 활용하시는 게 좋습니다.)


> 구체화 - 키오스크 동작

키오스크마다 주문 시 필요한 동작이 천차만별이다. 따라서 노인에게 적합한 키오스크 모니터 화면을 가상으로 구성하고, 각 동작(터치)을 제스처와 매칭하였다. 본 모니터 화면은 대한민국디자인전람회의 노인 키오스크 우수 디자인(모두 UI)을 참고하여 구성하였다.

### Part 3. 연구 계획
#### [사용 기술 및 언어]
Python, Visual Studio Code, OpenCV

#### [세부 일정]
연구 방향 설정 및 자료조사 : 09.13~ 09.25
1차 개발(손 검출 단계) : 09.26~ 10.09
2차 개발(제스처 인식 단계) : 10.10~ 10.23
중간 발표 : 10.24~ 10.30
3차 개발(결과물과 키오스크 화면 연동) : 11.07~ 11.13
실험 및 데모영상 제작 : 11.14~ 11.20
결과 분석(정확도, 소요 시간 등) : 11.21~ 11.27
발표자료 및 논문 작성 : 11.28~ 12.11
DeadLine : 12.14

### Part 4. 구현
#### [알고리즘]

### Part 5. 실험 및 결과분석
#### [실험]
노인을 위해 어떤 편의성을 제공해줄 수 있는지 고민해보고 그 편의성을 충족시키기 위해 무엇이 필요한지
#### [성능 평가]
정확도, 소요시간
### Part 6. 결론
#### [기대 효과]
#### [발전 가능성]

### Part 7. 참고문헌
> 논문

http://lps3.kiss.kstudy.com.sproxy.dongguk.edu/search/sch-result.asp

[Design of OpenCV based Finger Recognition System using binary processing and histogram graph](http://www.koreascience.or.kr/article/JAKO201608450941178.pdf)

[사용자 손 제스처 인식 기반 입체 영상 제어 시스템 설계 및 구현](https://koreascience.kr/article/JAKO202210858157166.pdf)

[웹캠을 이용한 동적 제스쳐 인식 기반의 감성 메신저 구현 및 성능분석](https://koreascience.kr/article/JAKO201030853097698.pdf)

[만 65세 이상 노인의 인지기능과 키오스크(KIOSK) 사용능력의 상관관계](https://kiss16-kstudy-com.sproxy.dongguk.edu/kiss61/download_viewer.asp)

[실제 인터뷰](https://kiss15-kstudy-com.sproxy.dongguk.edu/kiss5/download_viewer.asp)- 키오스크 서비스 실패: 시니어들의 부정적인 키오스크 이용 경험을 중심으로

[자연스런 손동작 인터렉션과 후마네트 운동을 접목한 노인성뇌질환 예방 기능성 게임기술](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002010995) - 손동작 선정 근거, 성능평가 참고

[노인 연령대별 손 기능 상관관계 연구](http://hanyang.dcollection.net/public_resource/pdf/200000436025_20221014141902.pdf)

[제스처인식 프레젠테이션](https://koreascience.kr/article/JAKO201126235931025.pdf)

> 오픈소스

[Rock Paper Scissors Machine](https://github.com/kairess/Rock-Paper-Scissors-Machine)
[Hand Gesture Recognition](https://github.com/kairess/gesture-recognition)

> 피드백
1. 키오스크 ui가 좌우로 쓸기의 경우 직관적이지 않음. -> 제스처 변경 또는 ui에 표시해주기
2. 숫자 3과 아래로 내리는(하, bad) 제스처는 노인이 하기에 적절하지 않아 보임. 
    -> 숫자 1,2,3 대신 숫자 0,1,2 / good, bad 대신 
3. 숫자 제스처가 다양한데 이를 다 허용할지, 특정 제스처만 되게 할지
    -> 모두 다 되게하고 펴진 손가락 개수를 인식하자. 왜? 
4. 키오스크에 카메라 위치에 따른 실험(제스처 인식  시 손 각도에 따라 인식률 실험)
5. 거리에 따라(노인의 경우 키오스크에서 매우 가까이 있을 것으로 보이므로) 인식률 실험 - 적어도 일정 거리 이상이 되어야 한다면 키오스크 앞에 일정 거리 라인을 세운다던지 해야 함.

