## Disorganized list:
airport security footage: https://www.youtube.com/watch?v=d9bViT8VEfc
 - has multiple lines, which isn't great

queue for bank security in news report: https://www.youtube.com/watch?v=iW8lQQzLDLc
 - Pans over queue pretty quickly
super long tsa line: https://www.youtube.com/watch?v=HsVTEoZjvwY
 - Huge amount of data to label
Line for mall: https://www.youtube.com/watch?v=VKM27gd1O7w
 - pretty good
helicopter view, low res: https://www.youtube.com/watch?v=7Xo69B9YQFc
types of people in a queue: https://www.youtube.com/watch?v=7Xo69B9YQFc
 - multiple good angles
 - focused on queues
Types of people in a queue 2: https://www.youtube.com/watch?v=SWV2swe2q4s
 drone footage for pooling station - defined line, small people on the whole: https://www.youtube.com/watch?v=qqyCKr-yGd0

queue for a kick at a taekwondo class: https://www.youtube.com/watch?v=AWUiqheLQjw
- straight side shot, seems pretty good

pretty good queue in the first few seconds: https://www.youtube.com/watch?v=DCYIM15wCCs

army troops in line - https://www.youtube.com/watch?v=Ym9EZqxrzHM
 - Almost everyone is in line, although not a huge number are

## Notes:
In general:
There were three general categories of videos videos explicitly of lines that seemed possibly relevant:
 1. Camera pans past *very* long line of people in it's entirety, or part of it - see the tsa video for a typical example
   - Most common type by a long shot
   - Transportation security lines are especially common
   - often came from news about how long the line was - think phone releases, voting day
   - Potentially too many people to label in a reasonable amount of time
   - Yolo could have trouble
   - Good candidate for having a distinction between people that are and are not in line
   - Often is at an angle that makes distinguishing people hard, i.e. close to parallel
 2. Aerial shots
   - Very clearly defined lines
   - Not great for yolo - potentially way too many people
 3. Side, fairly close (technically medium I think?) shots of lines
   - Clearly defined people
   - Tend to be perpendicular or close to it to the direction of the line
   - relatively small number of people
   - Often do not have anyone that is not in line - so it may not make sense as something try to use

Besides this, there are some edge cases like those from sports (esp. football lineups) or other situations where people are in a contrived order, but these cases aren't really in the spirit of the project as they aren't actually people in a queue

I think that group three is our best shot at actually getting the neural network to classify people, and will be by far the easiest to label. Group one is second best in this regard, and I'm fairly sure that group 2 isn't worth the effort. Again, group three has a major problem in that there typically aren't people *not* in line in these images, but I figure it would be best in terms of just trying to get something out the door by the deadline.
