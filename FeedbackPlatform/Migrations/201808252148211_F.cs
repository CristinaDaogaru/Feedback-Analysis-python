namespace FeedbackPlatform.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class F : DbMigration
    {
        public override void Up()
        {
            CreateTable(
                "dbo.Categories",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        Name = c.String(maxLength: 20),
                    })
                .PrimaryKey(t => t.Id);
            
            CreateTable(
                "dbo.Surveys",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        Name = c.String(maxLength: 20),
                        CategoryId = c.Int(nullable: false),
                        ClientId = c.Int(nullable: false),
                        QuestionId = c.Int(),
                    })
                .PrimaryKey(t => t.Id)
                .ForeignKey("dbo.ClientProfiles", t => t.ClientId, cascadeDelete: true)
                .ForeignKey("dbo.Questions", t => t.QuestionId)
                .ForeignKey("dbo.Categories", t => t.CategoryId, cascadeDelete: true)
                .Index(t => t.CategoryId)
                .Index(t => t.ClientId)
                .Index(t => t.QuestionId);
            
            CreateTable(
                "dbo.ClientProfiles",
                c => new
                    {
                        IdClient = c.Int(nullable: false, identity: true),
                        Name = c.String(nullable: false, maxLength: 20),
                        Email = c.String(nullable: false),
                        Password = c.String(nullable: false),
                    })
                .PrimaryKey(t => t.IdClient);
            
            CreateTable(
                "dbo.Questions",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        Description = c.String(),
                        SurveyId = c.Int(nullable: false),
                        Response_Id = c.Int(),
                    })
                .PrimaryKey(t => t.Id)
                .ForeignKey("dbo.Responses", t => t.Response_Id)
                .ForeignKey("dbo.Surveys", t => t.SurveyId)
                .Index(t => t.SurveyId)
                .Index(t => t.Response_Id);
            
            CreateTable(
                "dbo.Responses",
                c => new
                    {
                        Id = c.Int(nullable: false, identity: true),
                        QuestionId = c.Int(nullable: false),
                        Content = c.String(),
                        IsChecked = c.Boolean(nullable: false),
                    })
                .PrimaryKey(t => t.Id)
                .ForeignKey("dbo.Questions", t => t.QuestionId, cascadeDelete: true)
                .Index(t => t.QuestionId);
            
        }
        
        public override void Down()
        {
            DropForeignKey("dbo.Surveys", "CategoryId", "dbo.Categories");
            DropForeignKey("dbo.Questions", "SurveyId", "dbo.Surveys");
            DropForeignKey("dbo.Surveys", "QuestionId", "dbo.Questions");
            DropForeignKey("dbo.Responses", "QuestionId", "dbo.Questions");
            DropForeignKey("dbo.Questions", "Response_Id", "dbo.Responses");
            DropForeignKey("dbo.Surveys", "ClientId", "dbo.ClientProfiles");
            DropIndex("dbo.Responses", new[] { "QuestionId" });
            DropIndex("dbo.Questions", new[] { "Response_Id" });
            DropIndex("dbo.Questions", new[] { "SurveyId" });
            DropIndex("dbo.Surveys", new[] { "QuestionId" });
            DropIndex("dbo.Surveys", new[] { "ClientId" });
            DropIndex("dbo.Surveys", new[] { "CategoryId" });
            DropTable("dbo.Responses");
            DropTable("dbo.Questions");
            DropTable("dbo.ClientProfiles");
            DropTable("dbo.Surveys");
            DropTable("dbo.Categories");
        }
    }
}
