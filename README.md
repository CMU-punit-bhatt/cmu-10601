# [cmu-10601-intro-to-ml](http://www.cs.cmu.edu/~mgormley/courses/10601-f21/syllabus.html)
Syllabus
========

### Course Info

*   **Instructors**: [Matt Gormley](http://www.cs.cmu.edu/~mgormley/) and [Henry Chai](syllabus.html)
*   **Meetings**:
    *   **10-301 + 10-601 Section A:** MWF, 10:10 AM - 11:30 AM (Mellon Institute, Room 274K)
    *   **10-301 + 10-601 Section B:** MWF, 01:25 PM - 02:45 PM (CUC, McConomy Auditorium)
    *   **10-301 + 10-601 Section D:** Same times as Section B (online)
    *   For all sections, lectures are on Mondays and Wednesdays.
    *   Occasional recitations are on Fridays and will be announced ahead of time.
*   **Education Associates Email:** [eas-10-601@cs.cmu.edu](mailto:eas-10-601@cs.cmu.edu)
*   **Piazza:** [https://piazza.com/cmu/fall2021/1030110601/](https://piazza.com/cmu/fall2021/1030110601/)
*   **Gradescope:** [https://www.gradescope.com/courses/290314](https://www.gradescope.com/courses/290314)
*   **Lecture Livestream:** [See Piazza Post](https://piazza.com/class/ksg77m9s2cx3d6?cid=8)
*   **Video Recordings:** [https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=8f43dd60-b426-4a86-aabf-ad8a00f1e8d6](https://scs.hosted.panopto.com/Panopto/Pages/Sessions/List.aspx?folderID=8f43dd60-b426-4a86-aabf-ad8a00f1e8d6)
*   **Homework Handouts:** /10601-f21-website/coursework.html
*   **Office Hours Queue:** [https://cmu.ohqueue.com](https://cmu.ohqueue.com)

### 1\. Course Description

Machine Learning is concerned with computer programs that automatically improve their performance through experience (e.g., programs that learn to recognize human faces, recommend music and movies, and drive autonomous robots). This course covers the theory and practical algorithms for machine learning from a variety of perspectives. We cover topics such as Bayesian networks, decision tree learning, Support Vector Machines, statistical learning methods, unsupervised learning and reinforcement learning. The course covers theoretical concepts such as inductive bias, the PAC learning framework, Bayesian learning methods, margin-based learning, and Occam’s Razor. Programming assignments include hands-on experiments with various learning algorithms. This course is designed to give a graduate-level student a thorough grounding in the methodologies, technologies, mathematics and algorithms currently needed by people who do research in machine learning.

10-301 and 10-601 are identical. Undergraduates must register for 10-301 and graduate students must register for 10-601.

**Learning Outcomes:** By the end of the course, students should be able to:

*   Implement and analyze existing learning algorithms, including well-studied methods for classification, regression, structured prediction, clustering, and representation learning
*   Integrate multiple facets of practical machine learning in a single system: data preprocessing, learning, regularization and model selection
*   Describe the the formal properties of models and algorithms for learning and explain the practical implications of those results
*   Compare and contrast different paradigms for learning (supervised, unsupervised, etc.)
*   Design experiments to evaluate and compare different machine learning techniques on real-world problems
*   Employ probability, statistics, calculus, linear algebra, and optimization in order to develop new predictive models or learning methods
*   Given a description of a ML technique, analyze it to identify (1) the expressive power of the formalism; (2) the inductive bias implicit in the algorithm; (3) the size and complexity of the search space; (4) the computational properties of the algorithm: (5) any guarantees (or lack thereof) regarding termination, convergence, correctness, accuracy or generalization power.

For more details about topics covered, see the [Schedule](schedule.html) page.

### 2\. Prerequisites

Students entering the class are expected to have a pre-existing working knowledge of probability, linear algebra, statistics and algorithms, though the class has been designed to allow students with a strong numerate background to catch up and fully participate. In addition, recitation sessions will be held to review some basic concepts.

1.  You need to have, before starting this course, significant experience **programming** in a general programming language. Specifically, you need to have written from scratch programs consisting of several hundred lines of code. For undergraduate students, this will be satisfied for example by having passed 15-122 (Principles of Imperative Computation) with a grade of ‘C’ or higher, or comparable courses or experience elsewhere.
    
    Note: For each programming assignment, we will allow you to pick between **Python, C++, and Java**.
    
2.  You need to have, before starting this course, basic familiarity with **probability and statistics**, as can be achieved at CMU by having passed 36-217 (Probability Theory and Random Processes) or 36-225 (Introduction to Probability and Statistics I), or 15-359, or 21-325, or comparable courses elsewhere, with a grade of ‘C’ or higher.
    
3.  You need to have, before starting this course, college-level maturity in **discrete mathematics**, as can be achieved at CMU by having passed 21-127 (Concepts of Mathematics) or 15-151 (Mathematical Foundations of Computer Science), or comparable courses elsewhere, with a grade of ‘C’ or higher.
    

You must strictly adhere to these pre-requisites! Even if CMU’s registration system does not prevent you from registering for this course, it is still your responsibility to make sure you have all of these prerequisites before you register.

(Adapted from Roni Rosenfeld’s [10-601 Spring 2016](http://www.cs.cmu.edu/~roni/10601/) Course Policies.)

### 3\. Recommended Textbooks

*   [_Machine Learning_](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/mlbook.html), Tom Mitchell.
*   [_Machine Learning: a Probabilistic Perspective_](http://www.cs.ubc.ca/~murphyk/MLbook/), Kevin Murphy. Full online access is [free through CMU’s library](https://ebookcentral.proquest.com/lib/cm/detail.action?docID=3339490) – for the second link, you must be on CMU’s network or VPN.
*   [_A Course in Machine Learning_](http://ciml.info/), Hal Daumé III. Online only.

The core content of this course does not exactly follow any one textbook. However, several of the readings will come from the Murphy book (available free online via the library) and Daumé book (only available online). Some of the readings will include new chapters (available as free online PDFs) for the Mitchell book.

The textbook below is a great resource for those hoping to brush up on the prerequisite mathematics background for this course.

*   [_Mathematics for Machine Learning_](https://mml-book.github.io/), Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong. Free online.

### 4\. Course Components

#### Grading

The requirements of this course consist of participating in lectures, midterm and final exams, homework assignments, and readings. The grading breakdown is the following:

*   50% Homework Assignments
*   15% Exam 1
*   15% Exam 2
*   15% Exam 3 (during Final Exam week)
*   5% Participation
*   On Piazza, the _Top Student “Endorsed Answer” Answerers_ can earn bonus points

**Grade cutoffs:**

*   ≥ 97% A+
*   ≥ 93% A
*   ≥ 90% A-
*   ≥ 87% B+
*   ≥ 83% B
*   ≥ 80% B-
*   ≥ 77% C+
*   ≥ 73% C
*   ≥ 70% C-
*   ≥ 67% D+
*   ≥ 63% D
*   otherwise R

Each individual component (e.g. an exam) may be curved **upwards** at the end. As well, the cutoffs above are merely an **upper bound**, at the end they may be adjusted down. We expect that the number of students that receive A’s (including A+, A, A-) is at least half the number of students that take the midterm exam(s). The number of B’s (including B+, B, B-) will be at least two-thirds the number of A’s.

#### Midterm and Final Exams

Unless otherwise noted, all exams are closed-book.

You are required to attend the midterm and final exams. The midterm exam(s) will be given in the evening – not in class. The final exam will be scheduled by the registrar sometime during the official final exams period. Please plan your travel accordingly as we will not be able accommodate individual travel needs (e.g. by offering the exam early).

If you have an unavoidable conflict with an exam (e.g. an exam in another course), notify us **by filling out “exam conflict” form here:**

[Exam 1 Conflict Form (link TBD)](syllabus.html)

[Exam 2 Conflict Form (link TBD)](syllabus.html)

[Exam 3 Conflict Form (link TBD)](syllabus.html)

If your conflict is **with an exam in another course**, please promptly email the following people to let them know of the exam conflict:

*   the instructor(s) for this course
*   the education associate(s) for this course
*   the instructor(s) for the other course

In the case of exam conflicts, we will discuss with the other course and suggest a resolution. One possible resolution is that we will accommodate you and will ask you to fill out the exam conflict form above. Please note that email should ONLY be used to address conflicts with an exam in another course. The definition of a final exam conflict and the standard procedures are described here: [https://www.cmu.edu/hub/registrar/exams-and-grading/conflict-guidelines.html](https://www.cmu.edu/hub/registrar/exams-and-grading/conflict-guidelines.html)

#### Homework

The homeworks will divide into two types: programming and written. The programming assignments will ask you to implement ML algorithms from scratch; they emphasize understanding of real-world applications of ML, building end-to-end systems, and experimental design. The written assignments will focus on core concepts, “on-paper” implementations of classic learning algorithms, derivations, and understanding of theory.

More details are listed on the [Coursework](coursework.html) page.

#### Participation

##### In-Class Polls

We will be using Google Forms for in-class polls. Here’s how it will work:

1.  Sometime before the lecture, we will post a Google Form containing a few questions. In order to access it, _you must sign into Google using your Andrew Email_ – all students are automatically given access to [G Suite](https://www.cmu.edu/computing/services/comm-collab/collaboration/), which allows such a sign-in. The link to each poll will appear on the Schedule page.
2.  You will always be allowed to submit multiple times. So if there are multiple questions during a lecture, you should submit multiple times. You can do so by clicking the “Edit Your Response” link after each submission.
3.  _If you do not have a smartphone or tablet_, please pick up a poll card at the front of class as you enter and hand in a paper copy at the end of class. (Do _not_ submit a paper copy if you have a wireless device as it will create a mountain of paperwork for us.)

Here are some important notes regarding grading of these polls:

*   Each question will include a toxic option which if chosen will give you negative points. If you were to answer the polls randomly without learning the toxic option in class, you would receive negative points in expectation.
*   If you answer any non-toxic option for each question during class, you will receive full credit. If you answer any non-toxic option for each question after class within 24 hours of the end of the lecture, you will receive partial credit (50% credit).
*   Everyone receives 8 “free poll points” – meaning that you can skip up to 8 polls (~25% of lectures) and still get 100% for the in-class polls. As a result, you should never come to us asking for points because, e.g. your dog ate your smartphone. You cannot use more than 3 free polls consecutively! (Note that negative toxic points will consume multiple free polls.) Note that hitting a toxic option could easily wipe out 3 or more of your free poll points.

###### Exit Polls

Your submission of the exit polls counts towards your participation grade. There will be one after each homework assignment and one after each exam. You will receive full credit for any exit poll filled in within one week of its release on Piazza.

##### Mock Exams

Mock Exams are required. Before each real exam, we will ask you to complete an online, timed Mock Exam. The questions will probably be a bit easier than the real exams because they are designed to be autogradable, but we want you to have some low-stakes practice for the real exam. We expect you to put forth real effort on these Mock Exams and the points you receive therein will affect your participation grade.

##### Practice Problems

Practice Problems are optional. These problems will be released as a PDF. We will include the solutions for them as well. Some of these problems are exam-style and some are homework-style. The latter type are longer in form than we would typically include on an exam, but are still good practice.

#### Lectures

Attendance at lectures is expected except for those with explicit approval to miss lectures due to course conflicts or timezone conflicts. At least one section will be livestreamed and recorded for later viewing.

#### Recitations

Attendance at recitations (Friday sessions) is not required, but strongly encouraged. These sessions will be interactive and focus on problem solving; we strongly encourage you to actively participate. The recitations for at least one section will be livestreamed, and the recording will be available. A problem sheet will usually be released prior to the recitation. If you are unable to attend one or you missed an important detail, feel free to stop by office hours to ask the TAs about the content that was covered. Of course, we also encourage you to exchange notes with your peers.

#### Office Hours

The schedule for Office Hours will always appear on the Google Calendar on the People page.

#### Readings

The purpose of the readings is to provide a broader and deeper foundation than just the lectures and assessments. The readings for this course are **required**. We recommend you read them **after the lecture**. Sometimes the readings include whole topics that are not mentioned in lecture; such topics will (in general) **not** appear on the exams, but we still encourage you to skim those portions.

### 5\. Technologies

We use a variety of technologies:

#### Piazza

We will use Piazza for all **course discussion**. Questions about homeworks, course content, logistics, etc. should all be directed to Piazza. If you have a question, chances are several others had the same question. By posting your question publicly on Piazza, the course staff can answer once and everyone benefits. If you have a private question, you should also use Piazza as it will likely receive a faster response.

#### Gradescope

We use Gradescope to collect PDF submissions of **open-ended questions** on the homework (e.g. mathematical derivations, plots, short answers). The course staff will manually grade your submission, and you’ll receive personalized feedback explaining your final marks.

You will also submit your code for **programming questions** on the homework to Gradescope. After uploading your code, our grading scripts will autograde your assignment by running your program on a VM. This provides you with immediate feedback on the performance of your submission.

Exams will have a scheduled time and a fixed time limit, and will be live proctored in-person.

**Regrade Requests:** If you believe an error was made during manual grading, you’ll be able to submit a regrade request on Gradescope. For each homework, regrade requests will be open for only 1 week after the grades have been published. This is to encourage you to check the feedback you’ve received early!

#### Zoom

Lectures and recitations for at least one section will be livestreamed via Zoom.

#### Panopto

Lecture and recitation video recordings for at least one section will be available on Panopto. The link to the Video Recordings is available in the “Links” dropdown and the recitation recordings will be available.

### 6\. General Policies

#### Late homework policy

Late homework submissions are only eligible for 75% of the points the first day (24-hour period) after the deadline, 50% the second, and 25% the third.

You receive 6 total grace days **for use on any homework assignment except HW1**. We will automatically keep a tally of these grace days for you; they will be applied greedily. No assignment will be accepted more than 3 days after the deadline. This has two important implications: (1) you may not use more than 3 graces days on any single assignment (2) **you may not combine grace days with the late policy above to submit more than 3 days late.**

Only two grace days can be used on the written assignments (HW3, HW6, HW9). This ensures that we can promptly provide feedback to you before the subsequent exams.

All homework submissions are electronic (see Technologies section below). As such, lateness will be determined by the latest timestamp of any part of your submission. For example, suppose the homework requires two submission uploads – if you submit the first upload on time but the second upload 1 minute late, you entire homework will be penalized for the full 24-hour period.

#### Extensions

In general, we do not grant extensions on assignments. There are several exceptions:

*   **Medical Emergencies:** If you are sick and unable to complete an assignment or attend class, please go to University Health Services. For minor illnesses, we expect grace days or our late penalties to provide sufficient accommodation. For medical emergencies (e.g. prolonged hospitalization), students may request an extension afterwards.
*   **Family/Personal Emergencies:** If you have a family emergency (e.g. death in the family) or a personal emergency (e.g. mental health crisis), please contact your academic adviser and/or Counseling and Psychological Services (CaPS). In addition to offering support, they may reach out to the instructors for all your courses on your behalf to request an extension.
*   **University-Approved Travel:** If you are traveling out-of-town to a university approved event or an academic conference, you may request an extension for any time lost due to traveling. For university approved absences, you must provide confirmation of attendance, usually from a faculty or staff organizer of the event or via travel/conference receipts.

For any of the above situations, you may request an extension **by emailing the Education Associate(s) at [eas-10-601@cs.cmu.edu](mailto:eas-10-601@cs.cmu.edu) – do not email the instructor or TAs**. Please be specific about which assessment(s) you are requesting an extension for and the number of hours requested. The email should be sent as soon as you are aware of the conflict and at least **5 days prior to the deadline**. In the case of an emergency, no notice is needed.

If this is a medical emergency or mental health crisis, you must also CC your \[CMU College Liaison\] and your academic advisor. Do not submit any medical documentation to the course staff. If necessary, your College Liaison and The Division of Student Affairs (DoSA) will request such documentation and they will view the health documentation and conclude whether a retroactive extension is appropriate. (If you haven’t interacted with your college liaison before, they are experienced Student Affairs staff who work in partnership with students, housefellows, advisors, faculty, and associate deans in each college to assure support for students regarding their overall Carnegie Mellon experience.)

#### Audit Policy

Formal auditing of this course is permitted. However, we give priority to students taking the course for a letter grade. Auditors must register for one of the online sections.

You must follow the official procedures for a Course Audit as outlined by the HUB / registrar. Please do not email the instructor requesting permission to audit. Instead, you should first register for the appropriate section. Next fill out the [Course Audit Approval form](https://www.cmu.edu/hub/docs/course-audit.pdf) and obtain the instructor’s signature in-person (either at office hours or immediately after class).

Auditors are required to:

1.  Attend or watch all of the lectures.
2.  Receive a 95% participation grade or higher.
3.  Submit at least 3 of the 9 homework assignments.

Auditors are encouraged to sit for the exams, but should only do so if they plan to put forth actual effort in solving them.

#### Pass/Fail Policy

We allow you take the course as Pass/Fail. Instructor permission is not required. What letter grade is the cutoff for a Pass will depend on your specific program; we do not specify whether or not you Pass but rather we compute your letter grade the same as everyone else in the class (i.e. using the cutoffs listed above) and your program converts that letter grade to a Pass or Fail depending on their cutoff. Be sure to check with your program / department as to whether you can count a Pass/Fail course towards your degree requirements.

#### Accommodations for Students with Disabilities:

If you have a disability and have an accommodations letter from the Disability Resources office, I encourage you to discuss your accommodations and needs with me as early in the semester as possible. I will work with you to ensure that accommodations are provided as appropriate. If you suspect that you may have a disability and would benefit from accommodations but are not yet registered with the Office of Disability Resources, I encourage you to contact them at [access@andrew.cmu.edu](mailto:access@andrew.cmu.edu).

##### Remote Student Engagement

The remote-only (REO) Section for this course is only for students who cannot be on campus due to a government restriction on visas or travel or a documented health issue. To ensure that we are supporting these remote students:

*   Students in the remote section will be included in the course discussion forum, Piazza, to ask and answer questions.
*   Remote students will receive feedback on their assignments via marked rubric items in Gradescope, as all students do.
*   We encourage students to request office hours by appointment as needed. Please contact Joshmin Ray (joshminr@andrew.cmu.edu) if you need more support in office hours.
*   Lectures and recitations at the time of Section D will be livestreamed in real time. Remote students will be expected to participate in in-class polls during the normal class time. Recitations will be livestreamed, and the recordings will be available. Recordings for lectures will be available, but there may be a delay in posting these recordings while they are processed.
*   Exams will take place (remotely) at the same day and time as the in-person section. Accommodations for extreme time-zone differences will be considered on a case-by-case basis.

### 7\. Academic Integrity Policies

**Read this carefully!**

(Adapted from Roni Rosenfeld’s [10-601 Spring 2016](http://www.cs.cmu.edu/~roni/10601/) Course Policies.)

#### Collaboration among Students

*   The purpose of student collaboration is to facilitate learning, not to circumvent it. Studying the material in groups is strongly encouraged. It is also allowed to seek help from other students in understanding the material needed to solve a particular homework problem, provided any written notes (including code) are taken on an impermanent surface (e.g. whiteboard, chalkboard), and provided learning is facilitated, not circumvented. The actual solution must be done by each student alone.
*   A good method to follow when collaborating is to meet with your peers, discuss ideas at a high level, but do not copy down any notes from each other or from a white board. Any scratch work done at this time should be your own only. Before writing the assignment solutions, you should make sure that you are doing this without anyone else present, putting all notes away, closing all tabs on your computer, and writing it completely by yourself with no other resources.
*   You are absolutely not allowed to share/compare answers or screen share your work with one another.
*   There are two exceptions to the above policy:
    *   For each programming assignment (HW2, HW4, HW5, HW7, HW8), you will be randomly assigned to a homework group of 3. Homework groups will be different for each programming assignment. Within that homework group, you are permitted to show your code (either in-person or via Zoom screen share) to other members of your homework group. However, you may not write code while someone else’s code is being shared with you, i.e., you are not permitted to copy someone else’s code. You are certainly welcome to take mental notes, and learn from the design decisions they’ve made. All discussion and screen sharing between homework group members must happen either in person or on Zoom using your Andrew accounts by logging into http://cmu.zoom.us.
    *   For each written assignment (HW3, HW6, HW9), you will again be randomly assigned to a homework group of 3. These homework groups will also be different for each assignment. Every problem on each written assignment will be designated as SOLO or GROUP. Within your assigned homework group, you are allowed to share written notes pertaining to GROUP problems and you are allowed to write your solutions collectively.
*   The presence or absence of any form of help or collaboration, whether given or received, must be explicitly stated and disclosed in full by all involved. Specifically, each assignment solution must include answering the following questions:
    1.  Did you receive any help whatsoever from anyone in solving this assignment? Yes / No.
        *   If you answered ‘yes’, give full details: **\_\_****\_\_****\_\_****\_\_****\_\_****\_\_**
        *   (e.g. “Jane Doe explained to me what is asked in Question 3.4”)
    2.  Did you give any help whatsoever to anyone in solving this assignment? Yes / No.
        *   If you answered ‘yes’, give full details: **\_\_****\_\_****\_\_****\_\_****\_\_****\_\_**\_
        *   (e.g. “I pointed Joe Smith to section 2.3 since he didn’t know how to proceed with Question 2”)
    3.  Did you find or come across code that implements any part of this assignment ? Yes / No. (See below policy on “found code”)
        *   If you answered ‘yes’, give full details: **\_\_****\_\_****\_\_****\_\_****\_\_****\_\_**\_
        *   (book & page, URL & location within the page, etc.).
*   If you gave help after turning in your own assignment and/or after answering the questions above, you must update your answers before the assignment’s deadline, if necessary by emailing the course staff.
*   Collaboration without full disclosure will be handled severely, in compliance with [CMU’s Policy on Academic Integrity](https://www.cmu.edu/policies/student-and-student-life/academic-integrity.html).

#### Previously Used Assignments

Some of the homework assignments used in this class may have been used in prior versions of this class, or in classes at other institutions, or elsewhere. Solutions to them may be, or may have been, available online, or from other people or sources. It is explicitly forbidden to use any such sources, or to consult people who have solved these problems before. It is explicitly forbidden to search for these problems or their solutions on the internet. You must solve the homework assignments completely on your own. We will be actively monitoring your compliance. Collaboration with other students who are currently taking the class is allowed, but only under the conditions stated above.

#### Policy Regarding “Found Code”:

You are encouraged to read books and other instructional materials, both online and offline, to help you understand the concepts and algorithms taught in class. These materials may contain example code or pseudo code, which may help you better understand an algorithm or an implementation detail. However, when you implement your own solution to an assignment, you must put all materials aside, and write your code completely on your own, starting “from scratch”. Specifically, you may not use any code you found or came across. If you find or come across code that implements any part of your assignment, you must disclose this fact in your collaboration statement.

#### Duty to Protect One’s Work

Students are responsible for pro-actively protecting their work from copying and misuse by other students. If a student’s work is copied by another student, the original author is also considered to be at fault and in gross violation of the course policies. It does not matter whether the author allowed the work to be copied or was merely negligent in preventing it from being copied. When overlapping work is submitted by different students, both students will be punished.

To protect future students, do not post your solutions publicly, neither during the course nor afterwards.

#### Penalties for Violations of Course Policies

All violations (even first one) of course policies will always be reported to the university authorities (your Department Head, Associate Dean, Dean of Student Affairs, etc.) as an official Academic Integrity Violation and will carry severe penalties.

1.  The penalty for the first violation is a negative 100% on the assignment (i.e. it would have been better to submit nothing and receive a 0%).
    
2.  The penalty for the second violation is failure in the course, and can even lead to dismissal from the university.
    

### 8\. Support

Take care of yourself. Do your best to maintain a healthy lifestyle this semester by eating well, exercising, avoiding drugs and alcohol, getting enough sleep and taking some time to relax. This will help you achieve your goals and cope with stress.

All of us benefit from support during times of struggle. You are not alone. There are many helpful resources available on campus and an important part of the college experience is learning how to ask for help. Asking for support sooner rather than later is often helpful.

If you or anyone you know experiences any academic stress, difficult life events, or feelings like anxiety or depression, we strongly encourage you to seek support. Counseling and Psychological Services (CaPS) is here to help: call 412-268-2922 and visit their website at [http://www.cmu.edu/counseling/](http://www.cmu.edu/counseling/). Consider reaching out to a friend, faculty or family member you trust for help getting connected to the support that can help.

If you or someone you know is feeling suicidal or in danger of self-harm, call someone immediately, day or night:

*   CaPS: 412-268-2922
*   Re:solve Crisis Network: 888-796-8226
*   If the situation is life threatening, call the police:
    *   On campus: CMU Police: 412-268-2323
    *   Off campus: 911.

If you have questions about this or your coursework, please let the instructors know.

### 9\. Diversity

We must treat every individual with respect. We are diverse in many ways, and this diversity is fundamental to building and maintaining an equitable and inclusive campus community. Diversity can refer to multiple ways that we identify ourselves, including but not limited to race, color, national origin, language, sex, disability, age, sexual orientation, gender identity, religion, creed, ancestry, belief, veteran status, or genetic information. Each of these diverse identities, along with many others not mentioned here, shape the perspectives our students, faculty, and staff bring to our campus. We, at CMU, will work to promote diversity, equity and inclusion not only because diversity fuels excellence and innovation, but because we want to pursue justice. We acknowledge our imperfections while we also fully commit to the work, inside and outside of our classrooms, of building and sustaining a campus community that increasingly embraces these core values.

Each of us is responsible for creating a safer, more inclusive environment.

Unfortunately, incidents of bias or discrimination do occur, whether intentional or unintentional. They contribute to creating an unwelcoming environment for individuals and groups at the university. Therefore, the university encourages anyone who experiences or observes unfair or hostile treatment on the basis of identity to speak out for justice and support, within the moment of the incident or after the incident has passed. Anyone can share these experiences using the following resources:

*   Center for Student Diversity and Inclusion: csdi@andrew.cmu.edu, (412) 268-2150
*   Report-It online anonymous reporting platform: reportit.net username: tartans password: plaid

All reports will be documented and deliberated to determine if there should be any following actions. Regardless of incident type, the university will use all shared experiences to transform our campus climate to be more equitable and just.

### 10\. Research to Improve the Course

For this class, we are conducting research on teaching and learning. This research will involve some student work. You will not be asked to do anything above and beyond the normal learning activities and assignments that are part of this course. You are free not to participate in this research, and your participation will have no influence on your grade for this course or your academic career at CMU. If you do not wish to participate, please send an email to Chad Hershock (hershock@andrew.cmu.edu). Participants will not receive any compensation. The data collected as part of this research will include student grades. All analyses of data from participants’ coursework will be conducted after the course is over and final grades are submitted. The Eberly Center may provide support on this research project regarding data analysis and interpretation. The Eberly Center for Teaching Excellence and Educational Innovation is located on the CMU-Pittsburgh Campus and its mission is to support the professional development of all CMU instructors regarding teaching and learning. To minimize the risk of breach of confidentiality, the Eberly Center will never have access to data from this course containing your personal identifiers. All data will be analyzed in de-identified form and presented in the aggregate, without any personal identifiers. If you have questions pertaining to your rights as a research participant, or to report concerns to this study, please contact Chad Hershock (hershock@andrew.cmu.edu).

* * *

### 11\. Note to people outside CMU

Please feel free to reuse any of these course materials that you find of use in your own courses. We ask that you retain any copyright notices, and include written notice indicating the source of any materials you use.
